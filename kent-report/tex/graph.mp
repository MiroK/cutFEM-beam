u:=0.3cm;

beginfig(1)

draw fullcircle xscaled 2u yscaled 1.2u shifted (0u, 1u) withcolor black;
draw fullcircle xscaled 2u yscaled 1.2u shifted (-1.5u, -1u) withcolor black;
draw fullcircle xscaled 2u yscaled 1.2u shifted (1.5u, -1u) withcolor black;

label(btex Fem etex scaled 0.25, (0u, 1u));
label(btex Eigen etex scaled 0.25, (-1.5u, -1u));
label(btex Shen etex scaled 0.25, (1.5u, -1u));

drawarrow (-1.5u, -0.8u) -- (-0.1u, 0.8u) withcolor blue;
drawarrow (1.5u, -0.8u) -- (0.1u, 0.8u) withcolor red;
drawarrow (1.0u, -1u) -- (-1.0u, -1u) withcolor 0.5green dashed evenly;

endfig;
end;
