import numpy as np
from describe import describe_dataset

testdata = np.array([('Heidi Mitchell', 'uboyd@hotmail.com', 74, 52, 'female', '1121', 'cancer', '03/06/2018'),
       ('Tina Burns', 'stevenwheeler@williams.bi',  3, 86, 'female', '0323', 'hip', '26/09/2017'),
       ('Justin Brown', 'velasquezjake@gmail.com', 26, 56, 'female', '0100', 'heart', '31/12/2015'),
       ('Brent Parker', 'kennethsingh@strong-foley', 70, 57, 'male', '3131', 'heart', '02/10/2011'),
       ('Bryan Norton', 'erica36@hotmail.com', 48, 57, 'male', '0301', 'hip', '09/09/2012'),
       ('Ms. Erin Craig', 'ritterluke@gmail.com', 30, 98, 'male', '2223', 'cancer', '04/11/2006'),
       ('Gerald Park', 'larrylee@hayes-brown.net', 41, 73, 'female', '0101', 'heart', '15/12/2015'),
       ('Monica Fry', 'morenocraig@howard.com', 24,  1, 'male', '1212', 'hip', '21/12/2005'),
       ('Michael Smith', 'edward72@dunlap-jackson.c', 44, 66, 'male', '0111', 'hip', '07/11/2012'),
       ('Dean Campbell', 'michele18@hotmail.com', 62, 96, 'female', '2320', 'lung', '22/01/2009'),
       ('Kimberly Kent', 'wilsoncarla@mitchell-gree', 63, 51, 'male', '2003', 'cancer', '16/06/2017'),
       ('Michael Burnett', 'collin04@scott.org', 26, 88, 'male', '0301', 'heart', '07/03/2009'),
       ('Patricia Richard', 'deniserodriguez@hotmail.c', 94, 64, 'female', '3310', 'heart', '20/08/2006'),
       ('Joshua Ramos', 'michaelolson@yahoo.com', 59, 19, 'female', '3013', 'cancer', '22/07/2005'),
       ('Samuel Fletcher', 'jessicagarcia@hotmail.com', 14, 88, 'female', '1211', 'lung', '29/07/2004'),
       ('Donald Hess', 'rking@gray-mueller.com', 16, 15, 'male', '0102', 'hip', '16/09/2010'),
       ('Rebecca Thomas', 'alex57@gmail.com', 94, 48, 'female', '0223', 'cancer', '05/02/2000'),
       ('Hannah Osborne', 'ericsullivan@austin.com', 41, 25, 'female', '0212', 'heart', '11/06/2012'),
       ('Sarah Nelson', 'davidcruz@hood-mathews.co', 36, 57, 'female', '0130', 'cancer', '13/01/2003'),
       ('Angela Kelly', 'pwilson@howell-bryant.com', 37, 52, 'female', '1023', 'heart', '28/03/2009'),
       ('Susan Williams', 'smithjoshua@allen.com', 21, 42, 'male', '0203', 'lung', '15/11/2005')],
      dtype=[('name', '<U16'), ('email', '<U25'), ('age', '<i4'), ('weight', '<i4'), ('gender', '<U6'), ('zipcode', '<U6'), ('diagnosis', '<U6'), ('dob', '<U10')])

testdata2 = np.array([('Heidi Mitchell', 'uboyd@hotmail.com', 74, 52, 'female', '1121', 'cancer', '03/06/2018'),
       ('Tina Burns', 'stevenwheeler@williams.bi',  3, 86, 'female', '0323', 'hip', '26/09/2017'),
       ('Justin Brown', 'velasquezjake@gmail.com', 26, 56, 'female', '0100', 'heart', '31/12/2015'),
       ('Brent Parker', 'kennethsingh@strong-foley', 70, 57, 'male', '3131', 'heart', '02/10/2011'),
       ('Bryan Norton', 'erica36@hotmail.com', 48, 57, 'male', '0301', 'hip', '09/09/2012'),
       ('Ms. Erin Craig', 'ritterluke@gmail.com', 30, 98, 'male', '2223', 'cancer', '04/11/2006'),],
      dtype=[('name', '<U16'), ('email', '<U25'), ('age', '<i4'), 
             ('weight', '<i4'), ('gender', '<U6'), ('zipcode', '<U6'), 
             ('diagnosis', '<U6'), ('dob', '<U10')])

a=describe_dataset(testdata2, todescribe=['age'], condition=np.array(['f', 'f', 'f', 'm', 'm', 'm']))