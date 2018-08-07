# Barnes-Hut N-Body simulation
Implement the Barnes-Hut algorithm to compute the interaction of
a large number of attracting bodies.

The implementation is kinda-sorta parallel...

This was largely a test to see the result - the _physics_ are wonky
and broken and - newtonian. No measure of errors in the system exists,
and after a couple of frames it all is just mushy and has no structure.

Interesting however is the behavior of large clusters of bodies, they
seem to behave more like non-point masses (heck, they are LOTS of point
masses!). Also interesting is the local (and short lived) clumping
of particles all over the place.

High body-count _renderings_ can be found here:
* https://www.youtube.com/watch?v=BkjIVL-tc4g
* https://www.youtube.com/watch?v=63oeBsKPOxo
