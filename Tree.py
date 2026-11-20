import random
import math
import matplotlib
matplotlib.use('TKAgg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import agentpy as ap

class Tree(ap.Agent):
    def setup(self):
        # Species include Oak, Gum, and Wattle
        self.species = self.model.random.choice(['oak', 'gum', 'wattle'])
        # Age in days
        self.age = 0
        # Sunlight as a ratio from trees above to trees below. Calculated with biomass
        self.sunlight = 1.0
        # Stress (used to calculate mortality probability)
        self.stress = 0.0
        # Mortality Probability (Death to the Tree)
        self.mort_prob = 0.0
        self.is_dead = False
        # Seed Dispersal Parameters
        self.seed_dispersal_age = False
        self.seed_dispersal_prob = 0.0
        self.seed_process_counter = 0
        # DBH 
        self.dbh = 1
        # Stress Counters
        self.underwatered_counter = 0
        self.water_stress = 0
        self.temp_stress = 0
        self.light_stress = 0
        self.overage = 0
        # Hydraulic Roots
        self.hydro_roots = False
        
        if self.species == 'oak':
            # Researched by Mack, most shade-tolerant so 0.2
            self.light_dependency = 0.4
            # Taken from dataframe
            self.height = 10
            self.max_height = 2250
            self.max_dbh = 140
            # Data driven based on height
            self.biomass = 0.0
            # Water needed from Soil Moisture. Competitive Aspect
            self.water_dependency = 0.006
            # Heat Tolerance range (min,max) in Fahrenheit used to calculate stress
            self.heat_tolerance = (95, 100)
            self.age_threshold = 300 * 365
            self.hydro_roots = True
        elif self.species == 'gum':
            # Researched by Mack, least shade-tolerant so 0.6 (Full Sun)
            self.light_dependency = 0.8
            # Taken from dataframe
            self.height = 10
            self.max_height = 4000
            self.max_dbh = 40
            # Data driven based on height
            self.biomass = 0.0
            # Water needed from Soil Moisture. Competitive Aspect
            self.water_dependency = 0.013
            # Heat Tolerance range (min,max) in Fahrenheit used to calculate stress
            self.heat_tolerance = (112, 117)
            self.age_threshold = 500 * 365
        else: # This the Wattle 
            # Researched by Mack, moderately shade-tolerant so 0.4
            self.light_dependency = 0.65
            # Taken from dataframe
            self.height = 10
            self.max_height = 800
            self.max_dbh = 57
            # Data driven based on height
            self.biomass = 0.0
            # Water needed from Soil Moisture. Competitive Aspect
            self.water_dependency = 0.01
            # Heat Tolerance range (Min, Max) in Fahrenheit used to calculate stress
            self.heat_tolerance = (125,130)
            self.age_threshold = 30 * 365
    
    def growth(self, x_pos, y_pos):
        # Water Intake to Growth
        water_intake = min(self.model.grid.soil_moisture[x_pos,y_pos], self.water_dependency)
        self.water_taken = True if water_intake == self.water_dependency else False
        self.model.grid.soil_moisture[x_pos,y_pos] -= water_intake
        # Height Calculation (might add noise later)
        if water_intake < self.water_dependency and water_intake > 0:
            if self.hydro_roots and self.model.water_grid.water[x_pos, y_pos] != 0:
                if self.model.water_grid.water[x_pos, y_pos] > 0.4:
                    avail_water = max(self.model.water_grid.water[x_pos, y_pos], 0)
                    self.model.water_grid.water[x_pos, y_pos] -= avail_water
                    self.model.grid.soil_moisture[x_pos, y_pos] += avail_water
                else:
                    self.model.water_grid.water[x_pos, y_pos] -= 0.4
                    self.model.grid.soil_moisture[x_pos, y_pos] += 0.4
            else:
                self.height += 0
                self.dbh += 0
        elif water_intake == self.water_dependency:
            if self.species == 'gum':
                self.height += .584
                self.dbh += .033
                self.biomass = 0.0365 * (self.dbh ** 2.7324)- math.exp(-4.0813 + (5.8816/self.dbh))
            elif self.species == 'oak':
                self.height += .167
                self.dbh += .013
                self.biomass = (0.0000447 * (self.dbh ** 2.31958) * (self.height ** 0.62528))/(0.52)
            else:
                self.height += 0.233 # For the Wattle
                self.dbh += 0.022
                self.biomass = (0.0000904 * (self.dbh ** 2.18649) * (self.height ** 0.46736))/(0.48)
        else:
            self.height += 0
            self.dbh += 0
        
        # Making sure trees don't get taller
        if self.height > self.max_height:
            self.height = self.max_height
        
        if self.dbh > self.max_dbh:
            self.dbh = self.max_dbh
    
    def shading_stress(self):
        # Sunlight Values
        full_sun = .8
        mid_sun = .65
        low_sun = .3
        
        # Stress Levels
        A_level = 0.003
        B_level = 0.0075
        C_level = 0.01
        
        if self.sunlight < low_sun:
            if self.species == 'gum': # High Stress for Eucalyptus
                self.light_stress += C_level
            elif self.species == 'wattle': # Moderate Stress for Acacia
                self.light_stress += B_level
            else: # Low Stress for Quercus
                self.light_stress += A_level
        elif self.sunlight < mid_sun:
            if self.species == 'gum': # Moderate Stress for Eucalyptus
                self.light_stress += B_level
            elif self.species == 'wattle': # Low Stress for Acacia
                self.light_stress += A_level
            else: # Quercus Losing Stress
                self.light_stress -= A_level
        elif self.sunlight < full_sun:
            if self.species == 'gum': # Low Stress for Eucalyptus
                self.light_stress += A_level
            elif self.species == 'wattle': # Acacia Losing Stress
                self.light_stress -= B_level
            else: # Quercus Losing Stress
                self.light_stress -= A_level
        else:
            if self.species == 'gum': # Eucalyptus Losing Stress
                self.light_stress -= C_level
            elif self.species == 'wattle': # Acacia Losing Stress
                self.light_stress -= B_level
            else: # Quercus Losing Stress
                self.light_stress -= A_level
        
        if self.light_stress < 0:
            self.light_stress = 0
        elif self.light_stress >= 9:
            self.light_stress = 9
    
    def heat_stress(self):
        # Gaining Heat Stress
        if self.model.grid.cur_temp >= self.heat_tolerance[1]:
            self.temp_stress += 0.01
        elif self.model.grid.cur_temp >= self.heat_tolerance[0] and self.model.grid.cur_temp < self.heat_tolerance[1]:
            self.temp_stress += 0.003
        else:
            # Losing Heat Stress
            if self.temp_stress > 0:
                self.temp_stress = max(0, self.temp_stress - .01)
    
    def update_stress(self):
        # Calculating Age Threshold Stress
        if self.age >= self.age_threshold:
            self.overage = 1
            
        # Calculating Stress for Shading
        self.shading_stress()
        
        # Calculating Heat Stress
        self.heat_stress()
        
        # Gaining Water Stress
        if not self.water_taken:
            self.underwatered_counter += 1
        else:
            # Losing Water Stress
            if self.underwatered_counter > 0:
                self.underwatered_counter = max(0, self.underwatered_counter-1)
        # Calculationg Water Stress
        self.water_stress = self.underwatered_counter / 7
        
        # Calculating Overall Stress
        self.stress = min(9.0, max((self.water_stress + self.overage + self.temp_stress + self.light_stress), 0.0))
    
    def mortality(self):
        self.mort_prob = self.stress / 10
        if round(random.uniform(0.0, 1.0), 2) < self.mort_prob:
            self.is_dead = True
    
    def maturity(self):
        if self.age >= 365 and self.species == 'oak':
            self.seed_dispersal_age = True
            self.seed_dispersal_prob = 0.75
        elif self.age >= (365 * 2) and self.species == 'wattle':
            self.seed_dispersal_age = True
            self.seed_dispersal_prob = 0.75
        elif self.species == 'gum':
            if self.age >= (7 * 365):
                self.seed_dispersal_age = True
                self.seed_dispersal_prob = 0.75
            elif self.age >= (2*365):
                self.seed_dispersal_age = True
                self.seed_dispersal_prob = 0.15
            elif self.age >= (365//2):
                self.seed_dispersal_age = True
                self.seed_dispersal_prob = 0.05
    
    def seed_dispersal(self, x_pos, y_pos):
        if self.stress >= 3:
            self.seed_process_counter = 0
        else:
            self.seed_process_counter += 1
            if self.seed_process_counter == 14:
                if (n := random.uniform(0.0,1.0)) <= self.seed_dispersal_prob:
                    self.seed_process_counter = 0
                    poss = []
                    for x in range(-20, 21):
                        for y in range(-20, 21):
                            if x== 0 and y==0:
                                continue
                            
                            new_x = x_pos + x
                            new_y = y_pos + y
                            
                            if new_x < 0 or new_x > self.model.w:
                                continue
                            elif new_y < 0 or new_y > self.model.h:
                                continue
                            else:
                                pos = (new_x, new_y)
                                if pos in self.model.grid.empty:
                                    poss.append(pos)
                    if not poss:
                        return
                    
                    target_pos = self.model.random.choice(poss)
                    
                    seed = Tree(self.model)
                    seed.species = self.species
                    
                    self.model.trees.append(seed)
                    self.model.grid.add_agents([seed], positions=[target_pos])
    
    def step(self):
        if not self.is_dead:
            x, y = self.model.grid.positions[self]
            self.growth(x,y)
            self.age += 1
            self.update_stress()
            # To update chance of Eucalyptus
            if self.species == 'gum':
                self.maturity()
            # Seed Dispersing Behavior
            if self.seed_dispersal_age:
                self.seed_dispersal(x,y)
            else:
                self.maturity()
            # Mortality at End of Step
            if self.age % 7 == 0:
                if self.stress >= .5:
                    self.mortality()

class DroughtForestModel(ap.Model):
    def setup(self):
        self.w = self.p.width
        self.h = self.p.height
        n = self.p.trees
        self.min_temp = self.p.min_temp
        self.max_temp = self.p.max_temp
        
        # Grid created through parameters
        self.grid = ap.Grid(self, (self.w,self.h), track_empty = True)
        self.water_grid = ap.Grid(self, (self.w,self.h))
        # Each grid patch has this and I am saying that they represent 2 x 2 meter square
        self.grid.soil_moisture = np.ones((self.w,self.h))
        self.water_grid.water = np.ones((self.w,self.h))
        self.grid.light = np.ones((self.w,self.h))
        self.grid.rain_prob = self.p.rainfall
        self.grid.cur_temp = self.p.cur_temp
        # Tree List
        self.trees = ap.AgentList(self, n, Tree)
        self.tree_amt = n
        # Age Increasing
        for tree in self.trees:
            tree.age = 5475
            if tree.species == 'oak':
                self.height = min(tree.max_height, (.167*5475))
                self.dbh = min(tree.max_dbh, (.013*5475))
            elif tree.species == 'gum':
                self.height = min(tree.max_height, (.584*5475))
                self.dbh = min(tree.max_dbh, (.033*5475))
            else:
                self.height = min(tree.max_height, (.233*5475))
                self.dbh = min(tree.max_dbh, (.022*5475))
        # Added Trees to Grid
        self.grid.add_agents(self.trees, random=True, empty=True)
    
    def shading(self):
        sorted_trees = sorted(self.trees, key=lambda t: t.height, reverse=True)
        for tree in sorted_trees:
            neighbors = self.grid.neighbors(tree)
            taller_neighbors = [n for n in neighbors if n.height > tree.height and not tree.is_dead]
            
            if not taller_neighbors:
                tree.sunlight = 1.0
            else:
                candidates = [tree.biomass / n.biomass * n.sunlight for n in taller_neighbors]
                tree.sunlight = max(0.0, min(1.0, min(candidates)))
    
    def weather(self):
        min_t = 0
        max_t = 0
        if self.grid.cur_temp - 5 >= self.min_temp and self.grid.cur_temp + 5 <= self.max_temp:
            min_t = self.grid.cur_temp - 5
            max_t = self.grid.cur_temp + 5
        elif self.grid.cur_temp - 5 < self.min_temp:
            min_t = self.min_temp
            max_t = self.grid.cur_temp + (10 - (self.grid.cur_temp - self.min_temp))
        
        elif self.grid.cur_temp + 5 > 130:
            min_t = self.grid.cur_temp - (10 - (self.max_temp - self.grid.cur_temp))
            max_t = self.max_temp
        
        self.grid.cur_temp = random.randint(min_t, max_t)
    
    def step(self):
        self.grid.rain_prob = round(random.uniform(0.0, 0.1), 2)
        if round(random.uniform(0.0, 1.0), 2) < self.grid.rain_prob:
            self.grid.soil_moisture = np.ones((self.w,self.h))
            self.water_grid.water = np.ones((self.w,self.h))
        self.grid.soil_moisture = np.clip(self.grid.soil_moisture, 0.0, 1.0)
        self.weather()
        self.shading()
        self.trees.step()
        self.tree_amt = len(self.trees)
    
    def update(self):
        self.trees.record(['species', 'water_stress', 'temp_stress', 'light_stress', 'mort_prob', 'is_dead'])
        self.record('tree_amt')
    
    def end(self):
        self.report('final_tree_count', len(self.trees))

class ControlForestModel(ap.Model):
    def setup(self):
        self.w = self.p.width
        self.h = self.p.height
        n = self.p.trees
        self.min_temp = self.p.min_temp
        self.max_temp = self.p.max_temp
        
        # Grid created through parameters
        self.grid = ap.Grid(self, (self.w,self.h), track_empty = True)
        self.water_grid = ap.Grid(self, (self.w,self.h))
        # Each grid patch has this and I am saying that they represent 2 x 2 meter square
        self.grid.soil_moisture = np.ones((self.w,self.h))
        self.water_grid.water = np.ones((self.w,self.h))
        self.grid.light = np.ones((self.w,self.h))
        self.grid.rain_prob = self.p.rainfall
        self.grid.cur_temp = self.p.cur_temp
        # Tree List
        self.trees = ap.AgentList(self, n, Tree)
        for tree in self.trees:
            tree.species = self.p.tree_type
        self.tree_amt = n
        # Age Increasing
        for tree in self.trees:
            tree.age = 5475
            if tree.species == 'oak':
                self.height = min(tree.max_height, (.167*5475))
                self.dbh = min(tree.max_dbh, (.013*5475))
            elif tree.species == 'gum':
                self.height = min(tree.max_height, (.584*5475))
                self.dbh = min(tree.max_dbh, (.033*5475))
            else:
                self.height = min(tree.max_height, (.233*5475))
                self.dbh = min(tree.max_dbh, (.022*5475))
        # Added Trees to Grid
        self.grid.add_agents(self.trees, random=True, empty=True)
    
    def shading(self):
        sorted_trees = sorted(self.trees, key=lambda t: t.height, reverse=True)
        for tree in sorted_trees:
            neighbors = self.grid.neighbors(tree)
            taller_neighbors = [n for n in neighbors if n.height > tree.height and not tree.is_dead]
            
            if not taller_neighbors:
                tree.sunlight = 1.0
            else:
                candidates = [tree.biomass / n.biomass * n.sunlight for n in taller_neighbors]
                tree.sunlight = max(0.0, min(1.0, min(candidates)))
    
    def weather(self):
        min_t = 0
        max_t = 0
        if self.grid.cur_temp - 5 >= self.min_temp and self.grid.cur_temp + 5 <= self.max_temp:
            min_t = self.grid.cur_temp - 5
            max_t = self.grid.cur_temp + 5
        elif self.grid.cur_temp - 5 < self.min_temp:
            min_t = self.min_temp
            max_t = self.grid.cur_temp + (10 - (self.grid.cur_temp - self.min_temp))
        
        elif self.grid.cur_temp + 5 > 130:
            min_t = self.grid.cur_temp - (10 - (self.max_temp - self.grid.cur_temp))
            max_t = self.max_temp
        
        self.grid.cur_temp = random.randint(min_t, max_t)
    
    def step(self):
        self.grid.rain_prob = round(random.uniform(0.0, 0.1), 2)
        if round(random.uniform(0.0, 1.0), 2) < self.grid.rain_prob:
            self.grid.soil_moisture = np.ones((self.w,self.h))
            self.water_grid.water = np.ones((self.w,self.h))
        self.grid.soil_moisture = np.clip(self.grid.soil_moisture, 0.0, 1.0)
        self.weather()
        self.shading()
        self.trees.step()
        self.tree_amt = len(self.trees)
    
    def update(self):
        self.trees.record(['species', 'water_stress', 'temp_stress', 'light_stress', 'mort_prob', 'is_dead'])
        self.record('tree_amt')
    
    def end(self):
        self.report('final_tree_count', len(self.trees))


def animate_plot(model, ax):
    ax.clear()
    # Positions 
    positions = [model.grid.positions[a] for a in model.trees]
    heights = [a.height for a in model.trees]
    species = [a.species for a in model.trees]
    death = [a.is_dead for a in model.trees]

    if positions:
        xs, ys = zip(*positions)

        colors = []
        for i in range(len(species)):
            if death[i]:
                colors.append('black')
            elif species[i] == 'oak':
                colors.append('green')
            elif species[i] == 'gum':
                colors.append('red')
            else:
                colors.append('yellow')

        sizes = [min(30.0, h * 0.5) for h in heights]

        ax.scatter(xs, ys, s=sizes, c=colors, edgecolors='black')
        
    ax.set_title(f"Step {model.t} | Trees: {len(model.trees)}")
    ax.set_xlim(-0.5, model.p.width - 0.5)
    ax.set_ylim(-0.5, model.p.height - 0.5)

parameters = {
        'width': 50,
        'height': 50,
        'trees': 100,
        'rainfall': .1,
        'min_temp': 70,
        'max_temp': 130,
        'cur_temp': 110,
        'steps': 1825
    }

model = DroughtForestModel(parameters)
# fig, ax = plt.subplots(figsize=(7, 7))
# animation = ap.animate(model, fig, ax, plot=animate_plot)

# animation.save("forest.gif", writer="pillow", fps=4)

# exp = ap.Experiment(DroughtForestModel, parameters, record=True, iterations=30)
# results = exp.run()

# df1 = pd.DataFrame(results.variables['Tree'])
# df2 = pd.DataFrame(results.variables['DroughtForestModel'])
# df1.to_csv('results_Tree.csv')
# df2.to_csv('tree_amounts.csv')

# Control Groups
# Oak Control
parameters['tree_type'] = 'oak'
exp1 = ap.Experiment(ControlForestModel, parameters, record=True, iterations=30)
results_oak = exp1.run()

df3 = pd.DataFrame(results_oak.variables['Tree'])
df4 = pd.DataFrame(results_oak.variables['ControlForestModel'])
df3.to_csv('results_oak.csv')
df4.to_csv('oak_tree_amounts.csv')

# Eucalyptus Control
parameters['tree_type'] = 'gum'
exp2 = ap.Experiment(ControlForestModel, parameters, record=True, iterations=30)
results_gum = exp2.run()

df5 = pd.DataFrame(results_gum.variables['Tree'])
df6 = pd.DataFrame(results_gum.variables['ControlForestModel'])
df5.to_csv('results_gum.csv')
df6.to_csv('gum_tree_amounts.csv')

# Acacia Control 
parameters['tree_type'] = 'wattle'
exp3 = ap.Experiment(ControlForestModel, parameters, record=True, iterations=30)
results_wattle = exp3.run()

df7 = pd.DataFrame(results_wattle.variables['Tree'])
df8 = pd.DataFrame(results_wattle.variables['ControlForestModel'])
df7.to_csv('results_wattle.csv')
df8.to_csv('wattle_tree_amounts.csv')