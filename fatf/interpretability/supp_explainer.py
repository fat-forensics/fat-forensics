import numpy as np

testdata3 = np.array([
        ('Heidi Mitchell', 'uboyd@hotmail.com', 74, 52, 0, '0011', 1, '03/06/2018', 1),
       ('Tina Burns', 'stevenwheeler@williams.bi',  3, 86, 1, '0011', 0, '26/09/2017', 1),
       ('Justin Brown', 'velasquezjake@gmail.com', 3, 86, 2, '0011', 1, '31/12/2015', 0),
       ('Brent Parker', 'kennethsingh@strong-foley', 70, 57, 0, '0011', 1, '02/10/2011', 0),
       ('Bryan Norton', 'erica36@hotmail.com', 48, 57, 3, '1100', 0, '09/09/2012', 1),
       ('Ms. Erin Craig', 'ritterluke@gmail.com', 30, 98, 0, '1100', 1, '04/11/2006', 1),
       ('Gerald Park', 'larrylee@hayes-brown.net', 41, 73, 1, '1100', 0, '15/12/2015', 0),
       ],
      dtype=[('name', '<U16'), ('email', '<U25'), ('age', '<f4'), 
             ('weight', '<f4'), ('gender', '<i4'), ('zipcode', '<U6'), 
             ('target', '<i4'), ('dob', '<U10'), ('prediction', '<i4')])

def create_dataset():
    list_of_dictionaries = get_data()

    desired_keys = ['name',
                    'data',
                    'treatment',
                    'distance_func'
                    ]
    
    dts = []
    treatments = {
                'Protected': [],
                'Feature': [],
                'ToIgnore': [],
                'Target': []
                }
    distance_funcs = {}
    data = []
    
    
    for dictionary in list_of_dictionaries:
        current_dictionary_keys = dictionary.keys()
        for key in desired_keys:
            if key not in current_dictionary_keys:
                raise ValueError('One of the provided dictionaries does not have the key: ' + str(key))
        
        field_name = dictionary['name']
        field_col = dictionary['data']
        if type(field_col) != np.ndarray:
            raise TypeError(str(field_name) + ' data should be of type numpy.ndarray.')
        
        data.append(field_col)
        
        dts.append((field_name, field_col.dtype))
        distance_funcs[field_name] = dictionary['distance_func']

        field_treatment = dictionary['treatment']
        
        if field_treatment == 'Protected':
            treatments['Protected'].append(field_name)
        elif field_treatment == 'Feature':
            treatments['Feature'].append(field_name)
        elif field_treatment == 'Target':
            treatments['Target'].append(field_name)
        elif field_treatment == 'ToIgnore':
            treatments['ToIgnore'].append(field_name)
        else:
            raise ValueError('Unknown treatment')
            
    N = data[0].shape[0]
    if not np.all(column.shape[0] == N for column in data):
        raise ValueError('Data provided is of different length.')
        
    dataset = np.array([item for item in zip(*data)], dtype=dts)
    return dataset, treatments, distance_funcs

def get_dictionary(field_name, field_data, field_treatment, field_distance_func):
    dictionary =  {
                'name': field_name,
                'data': field_data,
                'treatment': field_treatment,
                'distance_func': field_distance_func
                }  
    
    return dictionary

def zipcode_dist(x, y):
    n = len(x)
    t = sum([item[0] == item[1] for item in zip(x, y)])
    return 1 - t/n
    
def get_data():
    age_dict = get_dictionary(field_name = 'Age', 
                              field_data = testdata3['age'], 
                              field_treatment = 'Feature', 
                              field_distance_func = lambda x, y: abs(x - y)
                              )
    
    weight_dict = get_dictionary(field_name = 'Weight', 
                              field_data = testdata3['weight'], 
                              field_treatment = 'Feature', 
                              field_distance_func = lambda x, y: abs(x - y)
                              )
    
    disease_dict = get_dictionary(field_name = 'Target', 
                              field_data = testdata3['target'],
                              field_treatment = 'Target', 
                              field_distance_func = lambda x, y: x == y
                              )


   
    zipcode_dict = get_dictionary(field_name = 'Zipcode', 
                              field_data = testdata3['zipcode'], 
                              field_treatment = 'Feature', 
                              field_distance_func = zipcode_dist
                              )
    
    gender_dict = get_dictionary(field_name = 'Gender', 
                              field_data = testdata3['gender'], 
                              field_treatment = 'Protected', 
                              field_distance_func = lambda x, y: x == y
                              )

    prediction_dict = get_dictionary(field_name = 'Prediction', 
                              field_data = testdata3['prediction'], 
                              field_treatment = 'ToIgnore', 
                              field_distance_func = None
                              )
    
    return [age_dict, weight_dict, disease_dict, gender_dict]
