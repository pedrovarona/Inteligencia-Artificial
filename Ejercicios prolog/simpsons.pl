% Hechos - Relaciones parentales
parent(homer, bart).
parent(homer, lisa).
parent(homer, maggie).
parent(marge, bart).
parent(marge, lisa).
parent(marge, maggie).
parent(abuela, marge).
parent(abuelo, homer).

% Hechos - Género
male(homer).
male(bart).
female(marge).
female(lisa).
female(maggie).
female(abuela).
male(abuelo).

% Reglas
ancestor(X, Y) :- parent(X, Y)
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
father(X, Y) :- parent(X, Y), male(X).
mother(X, Y) :- parent(X, Y), female(X).

grandmother(X, Y) :- female(X) , parent(X, Z), parent(Z, Y).
grandfather(X, Y) :- male(X) , parent(X, Z), parent(Z, Y).