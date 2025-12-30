% Hechos - Actores en películas
acted_in(leonardo, inception).
acted_in(leonardo, titanic).
acted_in(matthew, interstellar).

% Hechos - Directores de películas
directed_by(inception, nolan).
directed_by(interstellar, nolan).

% Regla - Actor versátil
versatile_actor(Actor) :- 
    acted_in(Actor, Movie),
    directed_by(Movie, nolan).

