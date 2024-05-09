import numpy
# np.savez_compressed('filename.npz', array1=array1, array2=array2)
# b = np.load('filename.npz')
b = numpy.load('/scratch/dr3631/neuro/mind-vis-main/data/Kamitani/npz/sbj_1.npz')
# for i in b.files:
#     print(i, b[i].shape)
print(b['V1'])