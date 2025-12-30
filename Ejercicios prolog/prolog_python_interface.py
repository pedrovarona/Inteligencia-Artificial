from pyswip import Prolog

prolog = Prolog()

prolog.consult("films.pl")
prolog.assertz("acted_in(matthew, aaaaa)")

results = prolog.query("versatile_actor(leonardo)")
print(len(list(results)) > 0)
print("*" * 20)
results = prolog.query("versatile_actor(asdfja)")
print(len(list(results)) > 0)

print("*" * 20)
for result in prolog.query("versatile_actor(X)"):
    print(result["X"])

print("*" * 20)
for result in prolog.query("acted_in(matthew, X)"):
    print(result["X"])
