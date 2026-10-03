# MHD Playground

**<https://roytsmart.github.io/mhd-playground/>**

Stir a magnetized fluid with your finger. Drag across the field lines and they
bend, then spring back as Alfvén waves; push oppositely directed field lines
together and they reconnect. It runs on your GPU with WebGL2, on a phone or a
computer.

## The physics

The simulation is two-dimensional, incompressible magnetohydrodynamics. In two
dimensions the magnetic field lies in the plane and is the curl of a single
flux function *A*:

*B* = (∂*A*/∂*y*, −∂*A*/∂*x*).

The field lines are the contours of *A*, which are the lines drawn, and the
field has no divergence by construction. With *J* = −∇²*A* the current out of
the plane, which the colors show,

∂*A*/∂*t* + *u*·∇*A* = *η*∇²*A*,

∂*u*/∂*t* + (*u*·∇)*u* = −∇*p* + *J*∇*A* + *ν*∇²*u*,  ∇·*u* = 0,

where *J*∇*A* is the Lorentz force *J* × *B*. The density is one, so the
Alfvén speed is |*B*|, and every starting point has its strongest field equal
to the field strength set on the page.

## The method

The fluid follows Stam's
["Stable Fluids"](https://doi.org/10.1145/311535.311548): semi-Lagrangian
advection of the velocity and of *A*, the Lorentz force, then a projection
onto divergence-free flow by Jacobi iteration for the pressure, all in
fragment shaders on a periodic grid over the unit square. *A* is stored as a
periodic part plus a uniform gradient, so that a uniform field can thread the
periodic grid.

Each frame, the GPU reduces the grid to its strongest field and fastest flow,
and the frame is split into substeps short enough that an Alfvén wave crosses
under half a cell and the flow carries the field lines at most a cell in each.
A quick stroke can squeeze the field lines together within a frame, so this is
measured every frame, after the fingers have pushed.

Reconnection happens through the resistivity *η* and through the grid's own
numerical diffusion, so its rate is qualitatively right rather than
quantitative. The flow is incompressible, so there are no shocks or sound
waves.

Two checks, made in the browser through the page's `window.mhd` testing
interface:

- A standing Alfvén wave of wavenumber 2π in a uniform field of strength one
  oscillates with period 1.0000, as it should, and loses less than half a
  percent of its amplitude in a period.
- A flux rope, an exact equilibrium, stays put, with stray flows under one
  percent of the Alfvén speed.

## Running it locally

It is a static page with no build step:

```bash
python -m http.server
```

and open <http://localhost:8000>. It needs WebGL2 and a GPU that can render to
floating-point textures, which almost every current phone and computer has.
