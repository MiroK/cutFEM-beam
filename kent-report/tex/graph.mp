u:=0.3cm;

beginfig(1)

draw fullcircle xscaled 2u yscaled 1.2u shifted (0u, 1u) withcolor black;
draw fullcircle xscaled 2u yscaled 1.2u shifted (-1.5u, -1u) withcolor black;
draw fullcircle xscaled 2u yscaled 1.2u shifted (1.5u, -1u) withcolor black;

label(btex Fem etex scaled 0.25, (0u, 1u));
label(btex Eigen etex scaled 0.25, (-1.5u, -1u));
label(btex Shen etex scaled 0.25, (1.5u, -1u));

path p;
p := (-1.5u, -0.8u){dir 135}..{dir 0}(-0.4u, 1u);
drawarrow  p withcolor blue;
pair v;
v = point 0.5 of p;
label(btex $\pi_m$ etex scaled 0.25, v shifted (-0.3u, 0u));

path pp;
pp := (1.5u, -0.8u){dir 45}..{dir 180}(0.4u, 1u);
drawarrow pp withcolor red;
pair vv;
vv = point 0.5 of pp;
label(btex $\Pi_m$ etex scaled 0.25, vv shifted (0.3u, 0u));

path ppp;
ppp := (1.0u, -0.9u){dir 135}..{dir -135}(-1.0u, -0.9u);
drawarrow ppp withcolor 0.5green dashed evenly;
pair vvv;
vvv = point 0.5 of ppp;
label(btex $\Pi_m$ etex scaled 0.25, vvv shifted (0u, 0.2u));

path pppp;
pppp := (-1.0u, -1.1u){dir -45}..{dir 45}(1.0u, -1.1u);
drawarrow pppp withcolor 0.5green dashed evenly;
pair vvvv;
vvvv = point 0.5 of pppp;
label(btex $\pi_m$ etex scaled 0.25, vvvv shifted (0u, -0.2u));
endfig;
end;
