import math

d = 10
age = 3650
euca_height = (.584 * age) + d
euca_dbh = min(.033 * age, 40)
euca_biomass = 0.0000763 * (euca_dbh ** 1.94553) * (euca_height ** 0.88389)/0.34

print('------- Eucalyptus --------')
print(f'Max Height: 4000, Height {euca_height}')
print(f'Max DBH: 40, DBH {euca_dbh}')
print(euca_biomass)
print()

age = 5475
oak_height = (.167 * 5475) + d
oak_dbh = .013 * 5475
oak_biomass = (0.0000447 * (oak_dbh ** 2.31958) * (oak_height ** 0.62528))/(0.52)

print('------- Oak --------')
print(f'Max Height: 2250, Height {oak_height}')
print(f'Max DBH: 140, DBH {oak_dbh}')
print(oak_biomass)
print()

wattle_height = 800
wattle_dbh = 57
wattle_biomass = (0.0000904 * (wattle_dbh ** 2.18649) * (wattle_height ** 0.46736))/(0.48)

print('------- Wattle --------')
print(f'Max Height: 800, Height {wattle_height}')
print(f'Max DBH: 57, DBH {wattle_dbh}')
print(wattle_biomass)
print()

print('-------- Sunlight -------')
oak_sunlight = oak_biomass/euca_biomass * 1
print(oak_sunlight)
wattle_sunlight = (wattle_biomass/oak_biomass * oak_sunlight)
print(wattle_sunlight)