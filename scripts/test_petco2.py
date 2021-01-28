from matplotlib import pyplot as plt

from vb_models_cvr.petco2 import CvrPetCo2Model

model = CvrPetCo2Model(None, phys_data='/Users/ctsu0221/data/cvr/Data/2018_119_001.txt')

plt.figure(1)
plt.plot(model.petco2_trim)
plt.plot(model.petco2_resamp, color='r')

plt.figure(2)
plt.plot(model.out_co2)
plt.show()

# %% write text files for the o2 and co2 EV's
# % fid=fopen('ev_co2.txt','Wt');
# % fprintf(fid,'%f \r',out_co2); 
# % fclose(fid);
# % 
# % fid=fopen('co2_tc.txt','Wt');
# % fprintf(fid,'%f \r',co2_tc); 
# % fclose(fid);
# % 
# % fid=fopen('end_tidals.txt','Wt');
# % fprintf(fid,'Normocapnia average in mmHg %f \r',normocap); 
# % fprintf(fid,'Hypercapnia value in mmHg %f \r',hypercap);
# % fclose(fid);
