Test for manual python env specification. Currently `envzy` dependency explorer can't find packages if they are
imported dynamically (i.e. inside functions, or even with `importlib` with dynamic name). This test checks that
user still can specify requirements himself and solve such problem.