#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <ctime>

#include <iterator>
#include <algorithm>
#include <vector>
#include <memory>

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

const unsigned Dimensions = 2;

// general constants to tweak computation of F
const auto ETA = 0.f; // settings this one higher leads to more "clumping"
const auto DIST_FACT = 8.f;
const auto G = 1e-4f; //6.6742e-11f;
const auto BETA = 0.5f;
const auto DT = 1.0f;

const auto max_coord = 1000.f;
const auto min_mass  = 1e2f;
const auto max_mass  = 1e6f;

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

const char *QuadName[] = {
   "NE",
   "NW",
   "SE",
   "SW"
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

      /* DBG(("{%f, %f}: got quadrant %s, selected offset {%f, %f}", d[0], d[1], QuadName[b], r[0], r[1])); */
      return r;
   }
};

Vec operator+(Vec a, Vec b) { return Vec{{a[0]+b[0], a[1]+b[1]}}; }
Vec operator-(Vec a, Vec b) { return Vec{{a[0]-b[0], a[1]-b[1]}}; }

Vec operator*(Vec a, flt b) { return Vec{{a[0]*b, a[1]*b}}; }
Vec operator/(Vec a, flt b) { return Vec{{a[0]/b, a[1]/b}}; }

Vec operator+=(Vec &a, Vec b) { a[0]+=b[0]; a[1]+=b[1]; return a; }
Vec operator-=(Vec &a, Vec b) { a[0]-=b[0]; a[1]-=b[1]; return a; }

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
//
// On tree construction:
//
// 1. Iterate the array of bodies and construct the tree whily doing so ...
// or
// 2. Start with a nil-tree and insert one bodie at a time, subdividing/creating
//    nodes where needed.
//
// 2 really sounds simple... and efficient too.

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
   flt               dt;

   bool              show_tree;
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
            flt x = frnd(size * 0.7f) + size * 0.01f;
            flt phi = flt(frnd(2 * M_PI));
            flt mass = frnd(max_mass - min_mass)+min_mass;

            Vec pos = Vec{{1, 0}};
            // body_count / 1000.f normalizes the whole thing to my testing
            // number of 1000 bodies.
            Vec vel = Vec{{0, rot * std::sqrt(G * mass * (body_count / 1000.f) * x / 30)}};

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

void populate_universe(Universe &u, size_t body_count)
{
   u = Universe();
   u.size = max_coord;
   u.dt = 0.05 * DT;
   u.show_tree = false;

   u.bodies.resize(body_count);

   std::vector<Body> a, b;
   create_galaxy(u, Vec{{-250,  250}}, Vec{{15,0}} * 4 * DIST_FACT * (body_count / 1e5f), 50, body_count / 4, a, 1.f);
   create_galaxy(u, Vec{{ 250, -250}}, Vec{{-7.5,0}} * 4 * DIST_FACT * (body_count / 1e5f), 300, body_count * 3 / 4, b, -1.f);

   u.bodies.swap(a);
   u.bodies.insert(u.bodies.end(), b.cbegin(), b.cend());

   /* std::for_each(u.bodies.begin(), u.bodies.end(), [u](Body &b) { }); */
}

void build_bhtree(Universe &u)
{
   u.nodes.clear();
   u.nodes.reserve(u.bodies.size() * 5 / 3);
   u.nodes.push_back(Node(Vec{{-u.size, -u.size}}, u.size*2));

   for (unsigned i = 0; i < u.bodies.size(); i++)
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

Vec compute_force(flt const x0, flt const y0, flt const m0,
                  flt const x1, flt const y1, flt const m1)
{
   // computation of F
   const auto dx = x1 - x0;
   const auto dy = y1 - y0;
   const auto r = 1.f / ((std::sqrt(dx * dx + dy * dy) + ETA) * DIST_FACT);
   const auto F = G * m0 * m1 * r;

   return Vec{{ F * dx * r, F * dy * r }};
}

Vec compute_force(Body const &i, Body const &j)
{
   return compute_force(i.pos[0], i.pos[1], i.mass, j.pos[0], j.pos[1], j.mass);
}

Vec compute_force(Body const &i, Node const &j)
{
   return compute_force(i.pos[0], i.pos[1], i.mass, j.center[0], j.center[1], j.mass);
}

void compute_acceleration(Body &i, Node const &j)
{
   const Vec F = compute_force(i, j);

   i.acc += F / i.mass;
}

void update_body(Universe &u, u32 q, u32 b, Vec const pos)
{
   if (u.nodes[q].body == b)
   {
      return;
   }

   if (u.nodes[q].body != Node::Internal)
   {
      compute_acceleration(u.bodies[b], u.nodes[q]);
      return;
   }

   const auto s = u.nodes[q].size * u.nodes[q].size;
   const auto dx = u.nodes[q].center[0] - pos[0];
   const auto dy = u.nodes[q].center[1] - pos[1];
   const auto d = dx * dx + dy * dy;

   if (s / d < BETA)
   {
      compute_acceleration(u.bodies[b], u.nodes[q]);
   }
   else
   {
      for (u32 i = 0; i < 4; i++)
      {
         if (u.nodes[q].childs[i])
         {
            update_body(u, u.nodes[q].childs[i], b, pos);
         }
      }
   }
}

void update_forces(Universe &u)
{
   for (u32 i = 0; i < u.bodies.size(); i++)
   {
      u.bodies[i].acc = Vec();
      update_body(u, 0, i, u.bodies[i].pos);
   }
}

void update_leapfrog(Universe &u)
{
   update_forces(u);

   const auto dt = u.dt;

   std::for_each(u.bodies.begin(), u.bodies.end(), [dt](Body &b) {
         b.pos += b.vel * 0.5f * dt;
         b.vel += b.acc * dt;
         b.pos += b.vel * 0.5f * dt;
   });
}

void update(Universe &u)
{
   build_bhtree(u);
   update_leapfrog(u);
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

               /* if (q.body == Node::Internal) */
               /*    return; */

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
   snprintf(buf, sizeof(buf), "Barnes-Hut %u :: Physics @ %0.2ffps", unsigned(uni->bodies.size()), 1e6f / t0);
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
      uni->dt *= 1.1f;
      printf("dt = %f\n", uni->dt);
      break;

   case '-':
      uni->dt *= 1.f / 1.1f;
      printf("dt = %f\n", uni->dt);
      break;

   case 't':
      uni->show_tree = !uni->show_tree;
      break;
   }
}

void run_glut(int argc, char **argv, Universe &u)
{
   uni = &u;

   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
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
   bh::Universe u;
   bh::populate_universe(u, 10000);

   run_glut(argc, argv, u);

   bh::depopulate_bhtree(u);

   return 0;
}
