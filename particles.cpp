#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <ctime>
#include <cstring>

#ifdef _OPENMP
#include <parallel/algorithm>
#else
#include <algorithm>
#endif

#include <iterator>
#include <vector>
#include <memory>

#include <pthread.h>

#include <unistd.h>

#ifdef USE_GLUT
#include <GL/glut.h>
#endif

#define DBG(X) (printf("DBG %s:%d: ", __FILE__, __LINE__), \
                printf X, \
                printf("\n"))

namespace bh
{

// Playing around with the barnes-hut algorithm to simulate a field of grvitating
// bodies. Each body has to account for all other bodies - hence the Barnes-Hut
// tree.
//
// What about using a binary tree to subdivide space? What about not subdividing
// space but the actual bodies (thing BVH or KD-Tree).

#if !defined(_OPENMP) && !defined(NO_THREADED_UPDATE)
const unsigned NTH = 8;
#endif

const auto G = 1.0e-4f;
const auto max_coord = 1000.f;

typedef std::uint32_t u32;
typedef float flt;

flt useconds()
{
   static double t0;
   if (! t0)
   {
      timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      t0 = ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
   }
   timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (ts.tv_sec * 1e6 + ts.tv_nsec / 1e3) - t0;
}

enum Quad
{
   NE,
   NW,
   SE,
   SW
};

struct Vec
{
   flt d[2];

   inline flt operator[](int i) const { return d[i]; }
   inline flt &operator[](int i) { return d[i]; }

   inline Vec operator()(u32 b) const {
      Vec r = Vec();

      switch (b) {
      case NE:
         r = Vec{{d[0], d[1]}};
         break;
      case NW:
         r = Vec{{0, d[1]}};
         break;
      case SE:
         r = Vec{{d[0], 0}};
         break;
      case SW:
         r = Vec{{0, 0}};
         break;
      }

      return r;
   }
};

inline Vec operator+(Vec const &a, Vec const &b) { return Vec{{a[0]+b[0], a[1]+b[1]}}; }
inline Vec operator-(Vec const &a, Vec const &b) { return Vec{{a[0]-b[0], a[1]-b[1]}}; }

inline Vec operator*(Vec const &a, flt b) { return Vec{{a[0]*b, a[1]*b}}; }
inline Vec operator/(Vec const &a, flt b) { return Vec{{a[0]/b, a[1]/b}}; }

inline Vec operator+=(Vec &a, Vec const &b) { a[0]+=b[0]; a[1]+=b[1]; return a; }
inline Vec operator-=(Vec &a, Vec const &b) { a[0]-=b[0]; a[1]-=b[1]; return a; }

inline flt dot(Vec const &a, Vec const &b) { return a[0]*b[0] + a[1]*b[1]; }

struct Body
{
   Vec pos;
   Vec vel;
   Vec acc;
   flt mass;
};

// Barnes-Hut uses a quad-tree for 2-dimensional simulations. As the tree
// subdivides space into even quadrants, irrespective of the contained bodies,
// the tree can get really high. That means we cannot use a heap-like structure
// of our tree (with a height of 16 we would need 4^16 == 2^32 nodes).

struct Node
{
   enum {
      Internal = u32(-1)
   };

   u32 childs[4];
   union {
      u32 bodies[3];
      u32 state;
   };
   Vec corner;
   Vec center;      // of mass
   flt mass;        // accumulated mass of all children
   flt size;        // edge length of the square
   u32 n;

   enum {
      NumBodies = sizeof(Node::bodies) / sizeof(Node::bodies[0])
   };

   inline Node()
      : childs()
      , bodies()
      , corner()
      , center()
      , mass()
      , size()
      , n()
   {
   }

   inline Node(Vec cornr, flt sz)
      : childs()
      , bodies()
      , corner(cornr)
      , center()
      , mass()
      , size(sz)
      , n()
   {
   }
};

struct Universe
{
   std::vector<Body> bodies;
   std::vector<Node> nodes;
   flt               size;

   struct Params
   {
      flt dt;
      flt beta;
      flt min_mass;
      flt max_mass;

      inline Params()
         : dt(0.25f)
         , beta(0.5f)
         , min_mass(1.0e2f)
         , max_mass(1.0e2f)
      {
      }
   };

   Params param;

   bool              show_tree;
   bool              bruteforce;
   bool              show_vel;
   bool              show_acc;

   struct Work {
      u32 id;
      pthread_t th;
      Universe *u;
   };

#if !defined(_OPENMP) && !defined(NO_THREADED_UPDATE)
   Work threads[NTH];
#endif

   inline Universe()
      : bodies()
      , nodes()
      , size(1000.f)
      , param()
      , show_tree()
      , bruteforce()
      , show_vel()
      , show_acc()
#if !defined(_OPENMP) && !defined(NO_THREADED_UPDATE)
      , threads()
#endif
   {
   }

   inline Universe(flt s, flt dt, flt beta)
      : bodies()
      , nodes()
      , size(s)
      , param()
      , show_tree()
      , bruteforce()
      , show_vel()
      , show_acc()
#if !defined(_OPENMP) && !defined(NO_THREADED_UPDATE)
      , threads()
#endif
   {
      param.dt = dt;
      param.beta = beta;
   }
};

inline flt frnd(flt max)
{
   float r = float(drand48());
   return r * max;
}

inline Quad quadrant_for_body(Universe const &u, Node const &q, u32 b, Vec &coff)
{
   const auto hsize = q.size * 0.5f;
   coff = Vec{{hsize, hsize}};
   const Vec center = q.corner + coff;

   const u32 west  = u.bodies[b].pos[0] < center[0];
   const u32 south = u.bodies[b].pos[1] < center[1];

   return Quad(west | (south << 1));
}

void bhtree_insert(Universe &u, u32 q, u32 b, int depth);

inline void bhtree_insert_next(Universe &u, u32 q, u32 b, int depth)
{
   Vec coff;
   Quad quad = quadrant_for_body(u, u.nodes[q], b, coff);

   if (! u.nodes[q].childs[quad])
   {
      u.nodes.push_back(Node(u.nodes[q].corner + coff(quad), u.nodes[q].size / 2));
      u.nodes[q].childs[quad] = u.nodes.size() - 1;
   }

   bhtree_insert(u, u.nodes[q].childs[quad], b, depth+1);
}

// this one really is ugly as hell... using ints all over to get a central "node" allcator
// in the universe... no fscken idea if it gives us an edge over a more naive aproach.
void bhtree_insert(Universe &u, u32 q, u32 b, int depth)
{
   int const max_depth = 1000;

   if (depth > max_depth) // .... wtf?!
   {
      DBG(("bhtree depth exceeded %d current q=%u, b=%u", max_depth, q, b));
      return;
   }

   if (u.nodes[q].state != Node::Internal && u.nodes[q].n < Node::NumBodies) // insert
   {
      const auto m = u.nodes[q].mass + u.bodies[b].mass;

      u.nodes[q].bodies[u.nodes[q].n++] = b;
      u.nodes[q].center = (u.nodes[q].center * u.nodes[q].mass + u.bodies[b].pos * u.bodies[b].mass) / m;
      u.nodes[q].mass = m;

      return;
   }
   
   if (u.nodes[q].state != Node::Internal) // leaf, need to subdivide and insert
   {
      for (u32 i = 0; i < u.nodes[q].n; i++)
         bhtree_insert_next(u, q, u.nodes[q].bodies[i], depth);
   }

   // update current node
   const auto m = u.nodes[q].mass + u.bodies[b].mass;
   u.nodes[q].center = (u.nodes[q].center * u.nodes[q].mass + u.bodies[b].pos * u.bodies[b].mass) / m;
   u.nodes[q].mass   = m;
   u.nodes[q].n      = 0;
   u.nodes[q].state  = Node::Internal;

   bhtree_insert_next(u, q, b, depth);
}

void create_galaxy(Universe &u, Vec center, Vec velocity, flt size, size_t body_count, std::vector<Body> &res, flt rot)
{
   res.reserve(res.size() + body_count);

   for (size_t i = 0; i < body_count; i++)
   {
      /* Vec pos; */
      flt x = frnd(size * 0.8f) + size * 0.001f;

      flt phi = flt(frnd(2 * M_PI));

      flt mass = frnd(u.param.max_mass - u.param.min_mass)+u.param.min_mass;

      Vec pos = Vec{{1, 0}};
      // body_count / 1000.f normalizes the whole thing to my testing
      // number of 1000 bodies.
      Vec vel = Vec{{0, rot * std::sqrt(G * mass * (body_count / 1000.f) * x)}};

      /* Vec vel = Vec{{0, std::sqrt(G * mass * (max_mass * body_count / 250000000.f))}}; */
      //Vec vel = Vec{{0, std::sqrt(G * (max_mass - min_mass) * body_count * 0.00036125f)}};

      Vec r = Vec{{std::cos(phi), std::sin(phi)}};
      Vec p = Vec{{pos[0]*r[0]-pos[1]*r[1],pos[0]*r[1]+pos[1]*r[0]}};
      Vec v = Vec{{vel[0]*r[0]-vel[1]*r[1],vel[0]*r[1]+vel[1]*r[0]}};

      pos = p * x + center;
      vel = v + velocity;

      res.push_back(Body{pos,
                         vel,
                         Vec(),
                         mass});
   }
}

void build_bhtree(Universe &u)
{
   u.nodes.clear();
   u.nodes.reserve(u.bodies.size() * 5 / 3);
   u.nodes.push_back(Node(Vec{{-u.size, -u.size}}, u.size*2));

   for (unsigned i = 0, iend = u.bodies.size(); i < iend; i++)
   {
      if (u.bodies[i].pos[0] < -u.size ||
          u.bodies[i].pos[1] < -u.size ||
          u.bodies[i].pos[0] >  u.size ||
          u.bodies[i].pos[1] >  u.size)
      {
         continue;
      }

      bhtree_insert(u, 0, i, 0);
   }
}

void depopulate_bhtree(Universe &u)
{
   u.nodes.clear();
}

inline void accelerate_body(Body &i, Vec const &j_pos, flt const j_mass)
{
   auto const d = j_pos - i.pos;
   auto const r = dot(d, d);
   auto const F = G * j_mass / r;

   // for momentum conservation would need to divide by i's mass, so
   // left it out completely.
   i.acc += d * F;
}

inline void update_body_acceleration(Body &i, Node const &j)
{
   accelerate_body(i, j.center, j.mass);
}

inline void update_body_acceleration(Body &i, Body const &j)
{
   accelerate_body(i, j.pos, j.mass);
}

void update_body(Universe &u, u32 q, u32 b, flt squared_beta)
{
   if (u.nodes[q].state != Node::Internal)
   {
      /* loop over bodies instead of using the node info, this is in order
       * to prevent adding our own contribution (if b is per chance in this node) */
      for (u32 i = 0; i < u.nodes[q].n; i++)
         if (u.nodes[q].bodies[i] != b)
            update_body_acceleration(u.bodies[b], u.bodies[u.nodes[q].bodies[i]]);
      return;
   }

   const auto s = u.nodes[q].size * u.nodes[q].size;
   const auto dv = u.nodes[q].center - u.bodies[b].pos;

   if (s / dot(dv, dv) < squared_beta)
   {
      update_body_acceleration(u.bodies[b], u.nodes[q]);
      return;
   }

   for (u32 i = 0; i < 4; i++)
   {
      if (u.nodes[q].childs[i])
      {
         update_body(u, u.nodes[q].childs[i], b, squared_beta);
      }
   }
}

void update_forces(Universe &u)
{
   u32 const iend = u.bodies.size();
   flt const squared_beta = u.param.beta * u.param.beta;
#ifdef _OPENMP
#pragma omp parallel for schedule(static,500)
#endif
   for (u32 i = 0; i < iend; i++)
   {
      u.bodies[i].acc = Vec();
      update_body(u, 0, i, squared_beta);
   }
}

#if !defined(_OPENMP) && !defined(NO_THREADED_UPDATE)
void *update_thread(void *data)
{
   Universe::Work &w = *static_cast<Universe::Work*>(data);
   Universe &u = *w.u;
   flt const squared_beta = u.param.beta * u.param.beta;

   for (u32 i = w.id, iend = u.bodies.size(); i < iend; i+=NTH)
   {
      u.bodies[i].acc = Vec();
      update_body(u, 0, i, squared_beta);
   }

   return 0;
}

void update_forces_threads(Universe &u)
{
   for (u32 t = 0; t < NTH; t++)
   {
      u.threads[t].id = t;
      u.threads[t].u = &u;
      int ret = pthread_create(&u.threads[t].th, NULL, update_thread, &u.threads[t]);
      if (ret != 0)
      {
         DBG(("Could not pthread_create() for id %u", t));
         std::abort();
      }
   }

   for (u32 t = 0; t < NTH; t++)
   {
      pthread_join(u.threads[t].th, NULL);
   }
}
#endif

void update_forces_brute(Universe &u)
{
   u32 const iend = u.bodies.size();
#ifdef _OPENMP
#pragma omp parallel for schedule(static,500)
#endif
   for (u32 i = 0; i < iend; i++)
   {
      u.bodies[i].acc = Vec();
      for(u32 j = 0; j < iend; j++)
         if (i != j)
            update_body_acceleration(u.bodies[i], u.bodies[j]);
   }
}

void update(Universe &u)
{
   if (u.bruteforce)
   {
      depopulate_bhtree(u);
      update_forces_brute(u);
   }
   else
   {
      build_bhtree(u);
#if !defined(_OPENMP) && !defined(NO_THREADED_UPDATE)
      update_forces_threads(u);
#else
      update_forces(u);
#endif
   }

   const auto dt = u.param.dt;

   // leapfrog integration
   std::for_each(u.bodies.begin(), u.bodies.end(), [dt](Body &b) {
         b.pos += b.vel * 0.5f * dt; // half dt psition update
         b.vel += b.acc * dt;        //      dt velocity update
         b.pos += b.vel * 0.5f * dt; // half dt position update with _new_ velocity
   });
}

void add_n_random(Universe &u, unsigned body_count, bool circle)
{
   for (unsigned i = 0; i < body_count; i++)
   {
      Vec pos = Vec{{frnd(2)-1, frnd(2)-1}} * u.size;

      if (circle)
      {
         while (pos[0]*pos[0] + pos[1]*pos[1] > u.size * u.size)
         {
            pos = Vec{{frnd(2)-1, frnd(2)-1}} * u.size;
         }
      }

      u.bodies.push_back(Body{
            pos,
            Vec(),
            Vec(),
            frnd(u.param.max_mass - u.param.min_mass) + u.param.min_mass
            });
   }
}

#ifdef USE_GLUT
static int width, height;
Universe *uni;

void show_bhtree(Universe &u)
{
   flt pxl_per_unit = 3.f * width / (u.size * 2.f);

   if (u.show_tree)
   {
      glColor3f(0.7f,1.0f,0.7f);
      std::for_each(u.nodes.cbegin(), u.nodes.cend(),
            [u](Node const &q) {
               if (q.size / width > 2.f)
                  return;

               flt v[2] = { q.corner[0], q.corner[1] };

               glBegin(GL_LINE_LOOP);
               glVertex2fv(v);
               v[0] += q.size;
               glVertex2fv(v);
               v[1] += q.size;
               glVertex2fv(v);
               v[0] -= q.size;
               glVertex2fv(v);
               glEnd();
            });
   }

   if (u.show_vel)
   {
      glColor3f(1,0,0);
      glBegin(GL_LINES);
      std::for_each(u.bodies.cbegin(), u.bodies.cend(),
            [pxl_per_unit](Body const &b) {
               glVertex2fv(b.pos.d);
               glVertex2fv((b.pos + b.vel * pxl_per_unit).d);
            });
      glEnd();
   }

   if (u.show_acc)
   {
      pxl_per_unit *= 10.f; // ... not sure if ...
      glColor3f(0,0,1);
      glBegin(GL_LINES);
      std::for_each(u.bodies.cbegin(), u.bodies.cend(),
            [pxl_per_unit](Body const &b) {
               glVertex2fv(b.pos.d);
               glVertex2fv((b.pos + b.acc * pxl_per_unit).d);
            });
      glEnd();
   }

   glColor3f(0,0,0);
   glEnableClientState(GL_VERTEX_ARRAY);
   glVertexPointer(2, GL_FLOAT, sizeof(Body), u.bodies.front().pos.d);
   glDrawArrays(GL_POINTS, 0, u.bodies.size());
}

void cb_display(void)
{
   glViewport(0, 0, width, height);

   glLoadIdentity();
   gluOrtho2D(-uni->size-1, uni->size, -uni->size-1, uni->size);

   glClearColor(1, 1, 1, 1);
   glClear(GL_COLOR_BUFFER_BIT);


   show_bhtree(*uni);

   glutSwapBuffers();
}

void cb_reshape(int w, int h)
{
   width = w;
   height = h;
}

void cb_idle(void)
{
   flt t0 = useconds();
   update(*uni);
   t0 = useconds() - t0;

   char buf[100];
   snprintf(buf, sizeof(buf), "%s %u :: dt=%f Physics @ %0.2ffps",
         uni->bruteforce ? "brute-force" : "Barnes-Hut",
         unsigned(uni->bodies.size()),
         uni->param.dt,
         1e6f / t0);
   glutSetWindowTitle(buf);

   glutPostRedisplay();

   /* static int frames = 0; */
   /* frames++; */
   /* if (frames == 1000) */
   /*    exit(0); */
}

void cb_keyboard(unsigned char k, int, int)
{
   switch (k) {
   case 'q': case 27:
      exit(0);
      break;

   case '+':
      uni->param.dt *= 1.1f;
      printf("dt = %f\n", uni->param.dt);
      break;

   case '-':
      uni->param.dt *= 1.f / 1.1f;
      printf("dt = %f\n", uni->param.dt);
      break;

   case 't':
      uni->show_tree = !uni->show_tree;
      break;

   case 'c':
      uni->bodies.clear();
      uni->nodes.clear();
      break;

   case 'v':
      uni->show_vel = !uni->show_vel;
      break;

   case 'a':
      uni->show_acc = !uni->show_acc;
      break;

   case 'h':
      add_n_random(*uni, 1000, true);
      break;

   case 'H':
      add_n_random(*uni, 1000, false);
      break;

   case 'b':
      uni->bruteforce = !uni->bruteforce;
      break;
   }
}

void run_glut(int argc, char **argv, Universe &u)
{
   uni = &u;

   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
   glutInitWindowSize(400, 400);
   glutCreateWindow("Barnes-Hut");

   glutDisplayFunc(cb_display);
   glutReshapeFunc(cb_reshape);
   glutKeyboardFunc(cb_keyboard);
   glutIdleFunc(cb_idle);

   glutMainLoop();

   bh::uni = nullptr;
}
#endif

void benchmark(Universe &u);

void make_universe(Universe &u, char **argv)
{
   // size <float>
   // beta <float>
   // dt <float>
   // min_mass <float>
   // max_mass <float>
   // scene n:unsigned "name"
   // galaxy n:<unsigned> size <float> pos x:<float> y:<float> vel x:<float> y:<float>
   // random [circle] body_count:<unsigned>

   u = Universe(1000.f, 0.05f, 0.5f);

   char const *two_galaxies[] = {
      "min_mass", "100", "max_mass", "100", "dt", "0.125", "size", "800",
      "galaxy", "2500", "pos", "0", "150", "vel", "4", "0", "size", "50", "rot", "-1",
      "galaxy", "7500", "pos", "0", "-300", "vel", "-1", "0", "size", "300", "rot", "-1",
      NULL
   };

   char const *galaxy[] = {
      "min_mass", "100", "max_mass", "100", "dt", "0.125", "size", "1000",
      "galaxy", "10000", "size", "1000",
      NULL
   };

   if (! *argv)
   {
      make_universe(u, (char**)two_galaxies);
      return;
   }

#define ifeq(x) if (std::strcmp(*argv, (x)) == 0 && printf("got %s\n", (x)))
#define elifeq(x) else ifeq(x)
#define atof(x) ([u](char const*a)->flt{ flt f = atof(a); printf("got %f\n", f); return f; })(x)
   while (*argv)
   {
      ifeq("dt") { u.param.dt = atof(*++argv); }
      elifeq("benchmark") {
         benchmark(u);
         exit(0);
      }
      elifeq("--help") {
         printf("Usage: particles [universe]\n\n");
         printf("Universe spec:\n");
         printf(
            "   size <float>\n"
            "   beta <float>\n"
            "   dt <float>\n"
            "   min_mass <float>\n"
            "   max_mass <float>\n"
            "   body mass:<float> [pos x:<float> y:<float>] [vel x:<float> y:<float>]\n"
            "   benchmark\n"
            "   scene <name>\n"
            "   galaxy n:<unsigned> size <float> [pos x:<float> y:<float>] [vel x:<float> y:<float>]\n"
            "   random [circle] body_count:<unsigned>\n");
         printf("\nExample Scenes:\n");
         char const **scenes[] = { galaxy, two_galaxies, NULL };
         char const *names[] = { "galaxy", "two-galaxies" };
         for (int i = 0; scenes[i]; i++)
         {
            printf("scene %s:\n  ", names[i]);
            for (int j = 0; scenes[i][j]; j++)
               printf("%s ", scenes[i][j]);
            printf("\n");
         }
         exit(1);
      }
      elifeq("size") { u.size = atof(*++argv); }
      elifeq("beta") { u.param.beta = atof(*++argv); }
      elifeq("max_mass") { u.param.max_mass = atof(*++argv); }
      elifeq("min_mass") { u.param.min_mass = atof(*++argv); }
      elifeq("scene") {
         argv++;
         ifeq("galaxy") { make_universe(u, (char**)galaxy); argv++; }
         elifeq("two-galaxies") { make_universe(u, (char**)two_galaxies); argv++; }
      }
      elifeq("body") {
         flt mass = 0.f;
         ifeq("random") { mass = frnd(u.param.max_mass - u.param.min_mass) + u.param.min_mass; }
         else mass = atof(*++argv);
         assert(mass > 0.f);
         argv++;
         Vec pos = Vec(), vel = Vec();
         while (*argv)
         {
            ifeq("pos") { pos[0] = atof(*++argv); pos[1] = atof(*++argv); }
            elifeq("vel") { vel[0] = atof(*++argv); vel[1] = atof(*++argv); }
            else { break; }
            argv++;
         }
         u.bodies.push_back(Body{pos, vel, Vec(), mass});
      }
      elifeq("galaxy") {
         char *endp;
         unsigned body_count = strtoul(*++argv, &endp, 10);
         assert(endp && "could not parse unsigned");
         if (endp) argv++;
         flt px = 0, py = 0, vx = 0, vy = 0;
         flt dia = 500.f;
         flt rot = 1.f;
         while (*argv)
         {
            ifeq("size") { dia = atof(*++argv); }
            elifeq("pos") { px = atof(*++argv); py = atof(*++argv); }
            elifeq("vel") { vx = atof(*++argv); vy = atof(*++argv); }
            elifeq("rot") { rot = atof(*++argv); }
            else { break; }
            argv++;
         }
         create_galaxy(u, Vec{{px, py}}, Vec{{vx, vy}}, dia, body_count, u.bodies, rot);
      }
      elifeq("random") {
         bool circle = false;
         argv++;
         ifeq("circle") { circle = true; } else { --argv; }
         unsigned body_count = strtoul(*++argv, NULL, 10);
         add_n_random(u, body_count, circle);
      }
      else
      {
         argv++;
      }
   }
#undef ifeq
#undef elifeq
#undef atof
}

void benchmark(Universe &u)
{
   flt t0 = useconds() / 1e3f;
   flt t1 = t0;
   for (int j = 0; j < 1000; j++)
   {
      flt t;
      if ((t = useconds() / 1e3f) - t1 > 100)
      {
         fprintf(stderr, ".");
         t1 = t;
      }
      update(u);
   }
   t0 = useconds() / 1e3f - t0;

   printf("\n%s %u :: dt=%f Physics @ %0.2ffps\n",
         u.bruteforce ? "brute-force" : "Barnes-Hut",
         unsigned(u.bodies.size()),
         u.param.dt,
         1e6f / t0);
}

} // namespace bh

int main(int argc, char **argv)
{
   bh::Universe u;

   bh::make_universe(u, argv+1);

#ifdef USE_GLUT
   run_glut(argc, argv, u);
#else
   bh::benchmark(u);
#endif

   return 0;
}
