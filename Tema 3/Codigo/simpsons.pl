% Hechos - Relaciones parentales
parent(homer, bart).
parent(homer, lisa).
parent(homer, maggie).
parent(marge, bart).
parent(marge, lisa).
parent(marge, maggie).

% Hechos - Género
male(homer).
male(bart).
female(marge).
female(lisa).
female(maggie).

% Reglas
ancestor(X, Y) :- parent(X, Y).
father(X, Y) :- parent(X, Y), male(X).
mother(X, Y) :- parent(X, Y), female(X).

