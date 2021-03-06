#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <ctime>
#include <cstring>
#include <cmath>

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

auto G = 1.0e-4f; //6.693e-11f; //1.0e-4f;
auto point_size = 2.f;
auto ETA = 3.f;
auto damp = 1.f;

typedef std::uint32_t u32;

#ifdef USE_DOUBLE
typedef double flt
#define GL_FLOAT GL_DOUBLE
#define glVertex2fv glVertex2dv
#else
typedef float flt;
#endif

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

float Q_rsqrt( float number )
{
   float x2;
   const float threehalfs = 1.5F;

   union {
      u32   i;
      float y;
   };

   x2 = number * 0.5F;
   y  = number;
   i  = 0x5f3759df - ( i >> 1 );               // what the fuck?
   y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
   y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed
   y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

   return y;
}

flt rsqrt(flt f)
{
#if defined(__x86__) || defined(__x86_64__)
   return 1.f / std::sqrt(f);
#else
   return Q_rsqrt(f);
#endif
}

inline Vec operator+(Vec const &a, Vec const &b) { return Vec{{a[0]+b[0], a[1]+b[1]}}; }
inline Vec operator-(Vec const &a, Vec const &b) { return Vec{{a[0]-b[0], a[1]-b[1]}}; }

inline Vec operator*(Vec const &a, flt b) { return Vec{{a[0]*b, a[1]*b}}; }
inline Vec operator/(Vec const &a, flt b) { return Vec{{a[0]/b, a[1]/b}}; }

inline Vec operator*=(Vec &a, flt b) { a[0]*=b; a[1]*=b; return a; }
inline Vec operator+=(Vec &a, Vec const &b) { a[0]+=b[0]; a[1]+=b[1]; return a; }
inline Vec operator-=(Vec &a, Vec const &b) { a[0]-=b[0]; a[1]-=b[1]; return a; }

inline flt dot(Vec const &a, Vec const &b) { return a[0]*b[0] + a[1]*b[1]; }

inline Vec normalize(Vec const &a) { return a * rsqrt(dot(a, a)); }

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
      u32 bodies[1];
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

struct View
{
   Vec pos;
   Vec dim;
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
      Universe *u;
   };

   enum { NumThreads = 8 };

   Work threads[NumThreads];
   pthread_barrier_t tb, tb2;

   std::vector<View> views;

   inline Universe()
      : bodies()
      , nodes()
      , size(1000.f)
      , param()
      , show_tree()
      , bruteforce()
      , show_vel()
      , show_acc()
      , threads()
      , tb()
      , tb2()
      , views{View{Vec{{-size, -size}}, Vec{{size, size}}*2}}
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
      , threads()
      , tb()
      , tb2()
      , views{View{Vec{{-size, -size}}, Vec{{size, size}}*2}}
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
   int const max_depth = 30000;

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
   res.reserve(res.size() + body_count + 1);

   const auto mult = 1e1f;
   const auto cm = body_count * u.param.max_mass * 0.5f * mult;
   res.push_back(Body{center, velocity, Vec(), cm});

   for (size_t i = 0; i < body_count; i++)
   {
      /* Vec pos; */
      flt xx = std::pow(frnd(1.f), 1.4142f);
      flt x = xx * size * 0.8f + size * 0.003f;

      flt phi = flt(frnd(2 * M_PI));

      xx = std::pow(frnd(1), 2.f);
      flt mass = xx*(u.param.max_mass - u.param.min_mass)+u.param.min_mass;

      Vec pos = Vec{{1, 0}};
      // body_count / 1000.f normalizes the whole thing to my testing
      // number of 1000 bodies.
      Vec vel = Vec{{0, rot * std::sqrt(G * cm / x)}};

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
   auto const r = dot(d, d) + ETA;
   auto const F = G * j_mass / r;

   i.acc += d * rsqrt(r) * F;
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

void *update_thread(void *data)
{
   Universe::Work &w = *(Universe::Work*) data;
   Universe &u = *w.u;

   while (1)
   {
      pthread_barrier_wait(&u.tb);

      auto const dt = u.param.dt;
      auto const squared_beta = u.param.beta * u.param.beta;
      auto const iend = u.bodies.size();

      auto const nwork = (iend + Universe::NumThreads - 1) / Universe::NumThreads;
      auto const wbegin = w.id * nwork;
      auto const wend = std::min((w.id + 1) * nwork, iend);

      for (auto i = wbegin; i < wend; i++)
      {
         Body &b = u.bodies[i];
         b.pos += b.vel * 0.5f * dt; // half dt psition update
      }
      pthread_barrier_wait(&u.tb2); // wai until all have finished the first step

      for (auto i = wbegin; i < wend; i++)
      {
         Body &b = u.bodies[i];
         b.acc = Vec();
         update_body(u, 0, i, squared_beta);
         b.vel += b.acc * dt;        //      dt velocity update
      }

      pthread_barrier_wait(&u.tb2); // wait until all have finished the sesond step
      for (auto i = wbegin; i < wend; i++)
      {
         Body &b = u.bodies[i];
         b.vel *= damp;
         b.pos += b.vel * 0.5f * dt; // half dt psition update
      }

      pthread_barrier_wait(&u.tb);
   }

   return nullptr;
}

void init_threading(Universe &u)
{
   static bool inited;

   if (! inited)
   {
      inited = true;

      pthread_barrier_init(&u.tb, NULL, Universe::NumThreads+1);
      pthread_barrier_init(&u.tb2, NULL, Universe::NumThreads);

      pthread_attr_t a;
      pthread_attr_init(&a);
      pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);

      for (int i = 0; i < Universe::NumThreads; i++)
      {
         u.threads[i].id = i;
         u.threads[i].u = &u;
         pthread_t t;
         pthread_create(&t, &a, update_thread, u.threads + i);
      }

      pthread_attr_destroy(&a);
   }
}

void update(Universe &u)
{
   build_bhtree(u);

#if !defined(_OPENMP) && !defined(NO_THREADED_UPDATE)
   init_threading(u);
   pthread_barrier_wait(&u.tb); // first sync for workers to start
   pthread_barrier_wait(&u.tb); // second sync for workers to finish
#else
   auto const dt = u.param.dt;
   auto const squared_beta = u.param.beta * u.param.beta;
   auto const iend = u.bodies.size();

   std::for_each(u.bodies.begin(), u.bodies.end(), [dt](Body &b) {
      b.pos += b.vel * 0.5f * dt; // half dt psition update
   });

#ifdef _OPENMP
#pragma omp parallel for schedule(static,500)
#endif
   for (auto i = 0u; i < iend; i++)
   {
      Body &b = u.bodies[i];
      b.acc = Vec();
      update_body(u, 0, i, squared_beta);
      b.vel += b.acc * dt;        //      dt velocity update
   }

   std::for_each(u.bodies.begin(), u.bodies.end(), [dt](Body &b) {
      b.vel *= damp;
      b.pos += b.vel * 0.5f * dt; // half dt position update with _new_ velocity
   });
#endif
}

void add_n_random(Universe &u, unsigned body_count, bool circle)
{
   u.bodies.reserve(u.bodies.size() + body_count);

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

void orthoProj(View const &v)
{
   Vec x = Vec{{v.pos[0], v.pos[0]+v.dim[0]}};
   Vec y = Vec{{v.pos[1], v.pos[1]+v.dim[1]}};

   glOrtho(x[0], x[1], y[0], y[1], 0, 1);
}

#ifdef USE_GLUT
static int width, height;
Universe *uni;

Vec drag[2];
bool dragging;

void show_bhtree(Universe &u)
{
   flt pxl_per_unit = 20 * (std::min(u.views.back().dim[0], u.views.back().dim[1]) * 2) / width;

   if (u.show_tree)
   {
      glColor3f(0.0f,0.2f,0.0f);
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

   glEnable(GL_BLEND);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

   if (u.show_vel)
   {
      glColor4f(1.f,0.3f,0.3f,0.5f);
      glBegin(GL_LINES);
      std::for_each(u.bodies.cbegin(), u.bodies.cend(),
            [pxl_per_unit](Body const &b) {
               glVertex2fv(b.pos.d);
               glVertex2fv((b.pos + normalize(b.vel) * pxl_per_unit).d);
            });
      glEnd();
   }

   if (u.show_acc)
   {
      glColor4f(0.3f,0.3f,1.f,0.5f);
      glBegin(GL_LINES);
      std::for_each(u.bodies.cbegin(), u.bodies.cend(),
            [pxl_per_unit](Body const &b) {
               glVertex2fv(b.pos.d);
               glVertex2fv((b.pos + normalize(b.acc) * pxl_per_unit).d);
            });
      glEnd();
   }

   glPointSize(point_size);

   glColor4f(1,1,1,0.2f);
   glEnableClientState(GL_VERTEX_ARRAY);
   glVertexPointer(2, GL_FLOAT, sizeof(Body), u.bodies.front().pos.d);
   glDrawArrays(GL_POINTS, 0, u.bodies.size());

   glDisable(GL_BLEND);

   if (dragging)
   {
      glColor4f(1,0,0,1);

      Vec v = drag[0];
      Vec dv = drag[1]-drag[0];

      glBegin(GL_LINE_LOOP);
      glVertex2fv(v.d);
      v[0] += dv[0];
      glVertex2fv(v.d);
      v[1] += dv[1];
      glVertex2fv(v.d);
      v[0] -= dv[0];
      glVertex2fv(v.d);
      glEnd();
   }
}

void cb_display(void)
{
   glViewport(0, 0, width, height);

   glLoadIdentity();
   orthoProj(uni->views.back());

   glClearColor(0, 0, 0, 0.3f);
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
   flt tf = 0, t0 = 0;
   int n = 0;

   /* while (tf + t0 < 1000000.f / 60.f) */
   {
      t0 = useconds();
      update(*uni);
      t0 = useconds() - t0;
      tf += t0;
      n++;
   }

   char buf[100];
   snprintf(buf, sizeof(buf), "%s %u :: dt=%f Physics @ %0.2ffps (%d)",
         uni->bruteforce ? "brute-force" : "Barnes-Hut",
         unsigned(uni->bodies.size()),
         uni->param.dt,
         1e6f / (tf / n), n);
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

   case '*':
      uni->param.dt *= -1.f;
      printf("dt %f\n", uni->param.dt);
      break;

   case '+':
      uni->param.dt *= 1.1f;
      printf("dt %f\n", uni->param.dt);
      break;

   case '-':
      uni->param.dt *= 1.f / 1.1f;
      printf("dt %f\n", uni->param.dt);
      break;

   case 't':
      uni->show_tree = !uni->show_tree;
      break;

   case 'C':
      uni->bodies.clear();
      uni->nodes.clear();
      printf("Cleared...\n");
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

   case 'G':
      {
         flt r = frnd(uni->size / 5.f) + uni->size / 20.f;
         Vec pos = Vec{{frnd(uni->size - r) - (uni->size - r) / 2, frnd(uni->size - r) - (uni->size - r) / 2}};
         Vec vel = Vec{{frnd(2)-1, frnd(2)-1}} * 0.5f;
         flt rot = frnd(1) < 0.5 ? -1 : 1;
         u32 body_count = u32(frnd(800)+200);
         printf("galaxy %u size %f pos %f %f vel %f %f rot %f\n", body_count, r, pos[0], pos[1], vel[0], vel[1], rot);
         create_galaxy(*uni, pos, vel, r, body_count, uni->bodies, rot);
      }
      break;
   }
}

Vec window_to_view(int x, int y)
{
   y = height - y;
   return Vec{{flt(x) / flt(width)  * uni->views.back().dim[0] + uni->views.back().pos[0],
               flt(y) / flt(height) * uni->views.back().dim[1] + uni->views.back().pos[1]}};
}

void cb_mouse(int button, int state, int x, int y)
{
   if (button == GLUT_LEFT_BUTTON)
   {
      auto const start = state == GLUT_DOWN;
      drag[!start] = window_to_view(x, y);
      printf("Translated window coords (%d, %d) to view (%f, %f)\n", x, y, drag[!start][0], drag[!start][1]);
      dragging = start;
      if (start)
      {
         drag[start] = drag[!start];
      }
      else
      {
         if (drag[0][0] > drag[1][0])
            std::swap(drag[0][0], drag[1][0]);
         if (drag[0][1] > drag[1][1])
            std::swap(drag[0][1], drag[1][1]);
         Vec d = drag[1] - drag[0];
         if (dot(d, d) > 0)
            uni->views.push_back(View{drag[0], drag[1] - drag[0]});
      }
   }
   else if (button == GLUT_RIGHT_BUTTON && state == GLUT_UP)
   {
      if (uni->views.size() > 1)
         uni->views.pop_back();
   }
}

void cb_motion(int x, int y)
{
   if (dragging)
   {
      drag[1] = window_to_view(x, y);
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
   glutMouseFunc(cb_mouse);
   glutMotionFunc(cb_motion);

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
      "min_mass", "100", "max_mass", "100", "dt", "0.0125", "size", "800",
      "galaxy", "1500", "pos", "0", "150", "vel", "0.3", "0", "size", "30", "rot", "-1",
      "galaxy", "7500", "pos", "0", "-150", "vel", "-0.15", "0", "size", "200", "rot", "-1",
      NULL
   };

   char const *galaxy[] = {
      "min_mass", "100", "max_mass", "100", "dt", "0.0125", "size", "1000",
      "galaxy", "10000", "size", "1000",
      NULL
   };

   char const *randm[] = {
      "size", "100", "min_mass", "1e8", "max_mass", "1e8", "G", "1e-9", "dt", "0.2", "eta", "20", "random", "circle", "10000", NULL
   };

   char const *solsys[] = {
      "G", "6.693e-11",
      "size", "235e9",

      // sun
      "body", "1.9891e30",
      // mercury
      "body", "3.302e23",    "pos", "57909175.0e3", "0",    "vel", "0", "47947.40069321917895390503",
      // venus
      "body", "4.8685e24",   "pos", "-108210000.0e3", "0",   "vel", "0", "-35075.6359725006440718531",
      // earth
      "body", "5.9736e24",   "pos", "0", "149597890.0e3",   "vel", "-29831.60633162660741273581", "0",
      // moon
      "body", "7.348e22",    "pos", "0", "149982295000",    "vel", "-30857.70412959226502854974", "0",
      // mars
      "body", "6.4185e23",   "pos", "0", "-227920000.0e3",   "vel", "24168.38180362685038388246", "0",

      NULL
   };

   if (! *argv)
   {
      make_universe(u, (char**)two_galaxies);
      return;
   }

#define ifeq(x) if (std::strcmp(*argv, (x)) == 0 && printf("got %s\n", (x)))
#define elifeq(x) else ifeq(x)
#define atof(x) ([u](char const*a)->flt{ flt f = atof(a); printf("got %g\n", f); return f; })(x)
   while (*argv)
   {
      ifeq("dt") { u.param.dt = atof(*++argv); }
      elifeq("G") { G = atof(*++argv); }
      elifeq("eta") { ETA = atof(*++argv); }
      elifeq("damp") { damp = atof(*++argv); }
      elifeq("point_size") { point_size = atof(*++argv); }
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
      elifeq("size") { u.size = atof(*++argv); u.views.back() = View{Vec{{-u.size, -u.size}}, Vec{{u.size, u.size}}*2}; }
      elifeq("beta") { u.param.beta = atof(*++argv); }
      elifeq("max_mass") { u.param.max_mass = atof(*++argv); }
      elifeq("min_mass") { u.param.min_mass = atof(*++argv); }
      elifeq("scene") {
         argv++;
         ifeq("galaxy") { make_universe(u, (char**)galaxy); argv++; }
         elifeq("random") { make_universe(u, (char**)randm); argv++; }
         elifeq("sol") { make_universe(u, (char**)solsys); argv++; }
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
