#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <ctime>

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

#include <GL/glut.h>

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

const unsigned NTH = 8;
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

   flt operator[](int i) const { return d[i]; }
   flt &operator[](int i) { return d[i]; }

   Vec operator()(u32 b) const {
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

Vec operator+(Vec const &a, Vec const &b) { return Vec{{a[0]+b[0], a[1]+b[1]}}; }
Vec operator-(Vec const &a, Vec const &b) { return Vec{{a[0]-b[0], a[1]-b[1]}}; }

Vec operator*(Vec const &a, flt b) { return Vec{{a[0]*b, a[1]*b}}; }
Vec operator/(Vec const &a, flt b) { return Vec{{a[0]/b, a[1]/b}}; }

Vec operator+=(Vec &a, Vec const &b) { a[0]+=b[0]; a[1]+=b[1]; return a; }
Vec operator-=(Vec &a, Vec const &b) { a[0]-=b[0]; a[1]-=b[1]; return a; }

flt dot(Vec const &a, Vec const &b) { return a[0]*b[0] + a[1]*b[1]; }

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
   enum State
   {
      Empty    = u32(-2),
      Internal = u32(-1)
   };

   u32 childs[4];
   Vec corner;
   Vec center;      // of mass
   flt mass;        // accumulated mass of all children
   u32 body;        // index of body or State
   flt size;        // edge length of the square

   Node()
      : childs()
      , corner()
      , center()
      , mass()
      , body(Empty)
      , size()
   {
   }

   Node(Vec cornr, flt sz)
      : childs()
      , corner(cornr)
      , center()
      , mass()
      , body(Empty)
      , size(sz)
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

      Params()
         : dt(0.25f)
         , beta(0.5f)
         , min_mass(1.0e2f)
         , max_mass(1.0e10f)
      {
      }
   };

   Params param;

   bool              show_tree;
   bool              bruteforce;

   struct Work {
      u32 id;
      pthread_t th;
      Universe *u;
   };

   Work threads[NTH];

   void (*scene)(Universe&, size_t);

   Universe()
      : bodies()
      , nodes()
      , size(1000.f)
      , param()
      , show_tree()
      , bruteforce()
      , threads()
      , scene()
   {
   }

   Universe(flt s, flt dt, flt beta, void (*scn)(Universe&,size_t))
      : bodies()
      , nodes()
      , size(s)
      , param()
      , show_tree()
      , bruteforce()
      , threads()
      , scene(scn)
   {
      param.dt = dt;
      param.beta = beta;
   }
};

flt frnd(flt max)
{
   float r = float(drand48());
   return r * max;
}

Quad quadrant_for_body(Universe const &u, Node const &q, u32 b, Vec &coff)
{
   const auto hsize = q.size * 0.5f;
   coff = Vec{{hsize, hsize}};
   const Vec center = q.corner + coff;

   const u32 west  = u.bodies[b].pos[0] < center[0];
   const u32 south = u.bodies[b].pos[1] < center[1];

   return Quad(west | (south << 1));
}

void bhtree_insert(Universe &u, u32 q, u32 b);

void bhtree_insert_next(Universe &u, u32 q, u32 b)
{
   Vec coff;
   Quad quad = quadrant_for_body(u, u.nodes[q], b, coff);

   if (! u.nodes[q].childs[quad])
   {
      u.nodes.push_back(Node(u.nodes[q].corner + coff(quad), u.nodes[q].size / 2));
      u.nodes[q].childs[quad] = u.nodes.size() - 1;
   }

   bhtree_insert(u, u.nodes[q].childs[quad], b);
}

// this one really is ugly as hell... using ints all over to get a central "node" allcator
// in the universe... no fscken idea if it gives us an edge over a more naive aproach.
void bhtree_insert(Universe &u, u32 q, u32 b)
{
   if (u.nodes[q].body == Node::Empty) // insert
   {
      u.nodes[q].body = b;
      u.nodes[q].mass = u.bodies[b].mass;
      u.nodes[q].center = u.bodies[b].pos;
      return;
   }

   if (u.nodes[q].body != Node::Internal) // leaf, need to subdivide and insert
   {
      bhtree_insert_next(u, q, u.nodes[q].body);
   }

   // update current node
   const auto m = u.nodes[q].mass + u.bodies[b].mass;
   u.nodes[q].center = (u.nodes[q].center * u.nodes[q].mass + u.bodies.at(b).pos * u.bodies.at(b).mass) / m;
   u.nodes[q].mass   = m;
   u.nodes[q].body   = Node::Internal;

   bhtree_insert_next(u, q, b);
}

void create_galaxy(Universe &u, Vec center, Vec velocity, flt size, size_t body_count, std::vector<Body> &res, flt rot)
{
   res.resize(body_count);
   std::generate(res.begin(), res.end(),
         [u, body_count, center, size, velocity, rot]() -> Body {
            /* Vec pos; */
            flt x = frnd(size * 0.8f) + size * 0.001f;

            flt phi = flt(frnd(2 * M_PI));

            flt mass = frnd(u.param.max_mass - u.param.min_mass)+u.param.min_mass;

            Vec pos = Vec{{1, 0}};
            // body_count / 1000.f normalizes the whole thing to my testing
            // number of 1000 bodies.
            Vec vel = Vec{{0, rot * std::sqrt(G * mass * (body_count / 1000.f) * x / 100)}};

            /* Vec vel = Vec{{0, std::sqrt(G * mass * (max_mass * body_count / 250000000.f))}}; */
            //Vec vel = Vec{{0, std::sqrt(G * (max_mass - min_mass) * body_count * 0.00036125f)}};

            Vec r = Vec{{std::cos(phi), std::sin(phi)}};
            Vec p = Vec{{pos[0]*r[0]-pos[1]*r[1],pos[0]*r[1]+pos[1]*r[0]}};
            Vec v = Vec{{vel[0]*r[0]-vel[1]*r[1],vel[0]*r[1]+vel[1]*r[0]}};

            pos = p * x + center;
            vel = v + velocity;

            return Body{pos,
                        vel,
                        Vec(),
                        mass};
         });
}

void populate_universe(Universe &u, size_t body_count, void (*scene)(Universe&, size_t))
{
   u = Universe(max_coord, 0.05f, 0.5f, scene);

   scene(u, body_count);

   /* std::for_each(u.bodies.begin(), u.bodies.end(), [u](Body &b) { }); */
}

void scene_two_galaxies(Universe &u, size_t body_count)
{
   u.param.min_mass = 1e2f;
   u.param.max_mass = 1e2f;
   u.param.dt = 1.f;

   std::vector<Body> a, b;

   create_galaxy(u, Vec{{0,  300}},
         Vec{{4,0}} * 2 / std::sqrt(40000.f / body_count),
         50,
         body_count / 4,
         a,
         -1.f);

   create_galaxy(u, Vec{{0, -300}},
         Vec{{-1,0}} * 2 / std::sqrt(40000.f / body_count),
         300,
         body_count * 3 / 4,
         b,
         -1.f);

   u.bodies.swap(a);
   u.bodies.insert(u.bodies.end(), b.cbegin(), b.cend());
}

void scene_galaxy(Universe &u, size_t body_count)
{
   u.param.min_mass = 1e2f;
   u.param.max_mass = 1e2f;
   u.param.dt = 0.25f;

   create_galaxy(u, Vec(),
         Vec(),
         u.size,
         body_count,
         u.bodies,
         -1.f);
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

      bhtree_insert(u, 0, i);
   }
}

void depopulate_bhtree(Universe &u)
{
   u.nodes.clear();
}

Vec compute_force(Vec const &a, Vec const &b, flt const m0, flt const m1)
{
   // computation of F
   const auto d = b - a;
   const auto r = 1.f / std::sqrt(dot(d, d));
   const auto F = G * m0 * m1 * r;

   return d * F * r;
}

void update_body_acceleration(Body &i, Node const &j)
{
   const Vec F = compute_force(i.pos, j.center, i.mass, j.mass);

   i.acc += F / i.mass;
}

void update_body_acceleration(Body &i, Body const &j)
{
   const Vec F = compute_force(i.pos, j.pos, i.mass, j.mass);

   i.acc += F / i.mass;
}

void update_body(Universe &u, u32 q, u32 b)
{
   if (u.nodes[q].body == b)
   {
      return;
   }

   if (u.nodes[q].body != Node::Internal)
   {
      update_body_acceleration(u.bodies[b], u.bodies[u.nodes[q].body]);
      return;
   }

   const auto s = u.nodes[q].size * u.nodes[q].size;
   const auto dv = u.nodes[q].center - u.bodies[b].pos;

   if (s / dot(dv, dv) < u.param.beta)
   {
      update_body_acceleration(u.bodies[b], u.nodes[q]);
   }
   else
   {
      for (u32 i = 0; i < 4; i++)
      {
         if (u.nodes[q].childs[i])
         {
            update_body(u, u.nodes[q].childs[i], b);
         }
      }
   }
}

void update_forces(Universe &u)
{
   u32 const iend = u.bodies.size();
#ifdef _OPENMP
#pragma omp parallel for schedule(static,500)
#endif
   for (u32 i = 0; i < iend; i++)
   {
      u.bodies[i].acc = Vec();
      update_body(u, 0, i);
   }
}

#if !defined(_OPENMP) && !defined(NO_THREADED_UPDATE)
void *update_thread(void *data)
{
   Universe::Work &w = *static_cast<Universe::Work*>(data);
   Universe &u = *w.u;

   for (u32 i = w.id, iend = u.bodies.size(); i < iend; i+=NTH)
   {
      u.bodies[i].acc = Vec();
      update_body(u, 0, i);
   }

   return 0;
}

void update_forces_threads(Universe &u)
{
   for (u32 t = 0; t < NTH; t++)
   {
      u.threads[t].id = t;
      u.threads[t].u = &u;
      pthread_create(&u.threads[t].th, NULL, update_thread, &u.threads[t]);
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

static int width, height;
Universe *uni;

void show_bhtree(Universe &u)
{
   if (u.show_tree)
   {
      glColor3f(0.7f,1.0f,0.7f);
      std::for_each(u.nodes.cbegin(), u.nodes.cend(),
            [u](Node const &q) {
               if (q.body == Node::Empty)
                  return;

               if (width / q.size > 200.f)
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

   glColor3f(0,0,0);
   glBegin(GL_POINTS);
   std::for_each(u.bodies.cbegin(), u.bodies.cend(),
         [](Body const &b) {
            glVertex2fv(b.pos.d);
         });
   glEnd();
}

void cb_display(void)
{
   glViewport(0, 0, width, height);

   glLoadIdentity();
   gluOrtho2D(-uni->size-1, uni->size, -uni->size-1, uni->size);

   glClearColor(1, 1, 1, 1);
   glClear(GL_COLOR_BUFFER_BIT);

   glPointSize(1.f);

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

   case 'R':
      {
         bool st = uni->show_tree;
         populate_universe(*uni, uni->bodies.size(), uni->scene);
         uni->show_tree = st;
      }
      break;

   case 'c':
      uni->bodies.clear();
      uni->nodes.clear();
      break;

   case 'h':
      {
         for (int i = 0; i < 1000; i++)
         {
            Vec pos;
            do {
               pos = Vec{{frnd(2)-1, frnd(2)-1}} * uni->size;
            } while (pos[0]*pos[0] + pos[1]*pos[1] > uni->size * uni->size);

            uni->bodies.insert(uni->bodies.end(), Body{
               pos,
               Vec(),
               Vec(),
               frnd(uni->param.max_mass - uni->param.min_mass) + uni->param.min_mass
            });
         }
      }
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
}

} // namespace bh

int main(int argc, char **argv)
{
   bh::u32 body_count = 5000;
   if (argv[1])
   {
      int bc = atoi(argv[1]);
      if (bc > 0 && bc < 1000000)
         body_count = bh::u32(bc);
   }

   bh::Universe u;
   bh::populate_universe(u, body_count, bh::scene_galaxy);
   /* bh::populate_universe(u, body_count, bh::scene_two_galaxies); */
   run_glut(argc, argv, u);

   return 0;
}
