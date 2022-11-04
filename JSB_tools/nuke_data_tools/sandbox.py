import re

sss="""self.element_name = data['Element']
            self.element_period = data['Period']
            self.element_group = data['Group']
            self.element_phase = data['Phase']
            self.metalQ = data['Metal'] == 'yes'
            self.nonmetalQ = data['Nonmetal'] == 'yes'
            self.metalloidQ = data['Metalloid'] == 'yes'
            self.element_type = data['Type']
            self.atomic_radius = data['AtomicRadius']
            self.electronegativity = data['Electronegativity']
            self.firstIonization = data['FirstIonization']
            self.density = data['Density']  # in g/cm3
            self.melting_point = data['MeltingPoint']
            self.boiling_point = data['BoilingPoint']
            self.specific_heat = data['SpecificHeat']
            self.n_valence_electrons = data['NumberofValence']"""

ls = []
for line in sss.split('\n'):
    s = line.split('=')[0]
    print(s)
    if m := re.match(" *self\.(.+)", s):
        ls.append(m.groups()[0])

print(ls)