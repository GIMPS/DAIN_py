import os.path as osp
import numpy as np

class Dataset(object):
    def __init__(self,root, split_id):
        self.root = root
        self.split_id = split_id
        self.train, self.val, self.train_val, self.test= [], [], [], []
        self.num_train_ids, self.num_val_ids = 0, 0
        self.num_class = 0
        self.material_label = {}
        self.split_path = osp.join(self.root, 'splits')
        self.load()

    def get_class_index(self):
        with open(osp.join(self.split_path, 'classInd.txt')) as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        f.close()
        for line in content:
            label,material_class = line.split(' ')
            if material_class not in self.material_label.keys():
                self.material_label[material_class]=label
        self.num_class = len(self.material_label)


    def load(self, num_val=0.3):
        self.get_class_index()
        self.train_val = self.file2_tuple_list(osp.join(self.split_path, 'trainlist0' + str(self.split_id) + '.txt'))

        # Randomly split train / val
        self.train_val = np.asarray(self.train_val)
        np.random.shuffle(self.train_val)
        num = len(self.train_val)
        if isinstance(num_val, float):
            num_val = int(round(num * num_val))
        if num_val >= num or num_val < 0:
            raise ValueError("num_val exceeds total identities {}"
                             .format(num))
        self.val = sorted(self.train_val[:-num_val])
        self.train = sorted(self.train_val[-num_val:])


        self.test = self.file2_tuple_list(osp.join(self.split_path, 'testlist0' + str(self.split_id) + '.txt'))
        self.num_train_ids = len(self.train)
        self.num_val_ids = len(self.val)
        self.num_test_ids = len(self.val)
        print ('==> (Training images, Validation images, Test images):(', self.num_train_ids,self.num_val_ids , self.num_test_ids,')')
        return self.train, self.val

    @property
    def images_dir(self):
        return osp.join(self.root, 'images')

    def file2_tuple_list(self, fname):
        with open(fname) as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        f.close()
        tl=[]
        for line in content:
            key = line.split('/',1)[1].split(' ',1)[0]
            label = int(self.material_label[line.split('/')[0]])-1
            tl.append((key,label))
        return tl
