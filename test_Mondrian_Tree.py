import unittest
import math
import random
import warnings

from Mondrian_Tree import Mondrian_Tree
from LeafNode import LeafNode


class test_Mondrian_Tree(unittest.TestCase):

    def setUp(self):

        self.d = 3
        self.n = 10
        self.a = 1
        self.linear_dims = [[0,1]]*self.d
        self.mt1 = Mondrian_Tree(self.linear_dims)

    # Basic tests

    def test_update_first_life_time_0p1(self):
        temp = Mondrian_Tree([[0,1]]*5)
        temp.update_life_time(0.1,set_seed=1)
        self.assertEqual(temp._num_leaves,3)

    def test_update_first_life_time_0p5(self):
        temp = Mondrian_Tree([[0,1]]*5)
        temp.update_life_time(0.5,set_seed=1)
        self.assertEqual(temp._num_leaves,8)

    def test_update_first_life_time_1(self):
        temp = Mondrian_Tree([[0,1]]*5)
        temp.update_life_time(1,set_seed=1)
        self.assertEqual(temp._num_leaves,36)

    # Testing make leaf list 

    def test_make_full_leaf_list_base(self):
        self.mt1.make_full_leaf_list()
        self.assertEqual(len(self.mt1._full_leaf_list),1)

    def test_make_full_leaf_list(self):
        lbda = 1
        self.mt1.update_life_time(lbda,set_seed=1)
        self.mt1.make_full_leaf_list()
        # print(len(self.mt1._full_leaf_list))
        # for node in self.mt1._full_leaf_list:
        #     print(node.leaf_id)
        self.assertEqual(len(self.mt1._full_leaf_list),self.mt1._num_leaves)

    # Testing input data

    def test_input_data_empty_root(self):
        self.mt1.input_data([],[],[])
        self.assertEqual(self.mt1._num_points,0)

    def test_input_data_empty(self):
        lbda = 1
        self.mt1.update_life_time(lbda, set_seed=1)
        self.mt1.input_data([],[],[])
        self.assertEqual(self.mt1._num_points,0)

    def test_input_data_root_no_labels(self):
        self.mt1.input_data([[0]*self.d],[],[])
        self.assertEqual(self.mt1._num_points,1)
        self.assertEqual(self.mt1.labels,[None])
        self.assertEqual(self.mt1._root.unlabelled_index,[0])
        self.assertEqual(self.mt1._root.labelled_index,[])

    def test_input_data_root_labels(self):
        self.mt1.input_data([[0]*self.d],[0],[3.141])
        self.assertEqual(self.mt1._num_points,1)
        self.assertEqual(self.mt1.labels,[3.141])
        self.assertEqual(self.mt1._root.unlabelled_index,[])
        self.assertEqual(self.mt1._root.labelled_index,[0])

    def test_input_data(self):
        d = 3
        lbda = 0.5
        n_points = 100
        n_labelled = 20
        temp_tree = Mondrian_Tree([[0,1]]*d)
        temp_tree.update_life_time(lbda, set_seed=100)

        labelled_indicies = range(n_labelled)
        labels = [1]*n_labelled
        data = []
        random.seed(1)
        for i in range(n_points):
            point = []
            for j in range(d):
                point.append(random.random())
            data.append(point)
        temp_tree.input_data(data, labelled_indicies, labels)

        self.assertEqual(temp_tree._num_points,n_points)
        self.assertEqual(temp_tree._num_labelled,n_labelled)
        
        temp_tree.make_full_leaf_list()
        for node in temp_tree._full_leaf_list:
            # print(len(node.labelled_index), len(node.unlabelled_index))

            linear_dims = node.linear_dims
            for ind in node.labelled_index:
                point = temp_tree.points[ind]
                for dim in range(d):
                    # print(linear_dims[dim][0], point[dim], linear_dims[dim][1])
                    self.assertTrue(point[dim] >= linear_dims[dim][0])
                    self.assertTrue(point[dim] <= linear_dims[dim][1])

            for ind in node.unlabelled_index:
                point = temp_tree.points[ind]
                for dim in range(d):
                    # print(linear_dims[dim][0], point[dim], linear_dims[dim][1])
                    self.assertTrue(point[dim] >= linear_dims[dim][0])
                    self.assertTrue(point[dim] <= linear_dims[dim][1])

    def test_update_life_time_with_data(self):

        d = 3
        lbda = 0.5
        lbda2 = 1
        n_points = 100
        n_labelled = 20
        temp_tree = Mondrian_Tree([[0,1]]*d)
        temp_tree.update_life_time(lbda, set_seed=100)
        # print(temp_tree._num_points)

        labelled_indicies = range(n_labelled)
        labels = [1]*n_labelled
        data = []
        random.seed(1)
        for i in range(n_points):
            point = []
            for j in range(d):
                point.append(random.random())
            data.append(point)
        temp_tree.input_data(data, labelled_indicies, labels)
        temp_tree.update_life_time(lbda2, set_seed=100)
        # print(temp_tree._num_points)

        temp_tree.make_full_leaf_list()
        for node in temp_tree._full_leaf_list:
            # print(len(node.labelled_index), len(node.unlabelled_index))

            linear_dims = node.linear_dims
            for ind in node.labelled_index:
                point = temp_tree.points[ind]
                for dim in range(d):
                    # print(linear_dims[dim][0], point[dim], linear_dims[dim][1])
                    self.assertTrue(point[dim] >= linear_dims[dim][0])
                    self.assertTrue(point[dim] <= linear_dims[dim][1])

            for ind in node.unlabelled_index:
                point = temp_tree.points[ind]
                for dim in range(d):
                    # print(linear_dims[dim][0], point[dim], linear_dims[dim][1])
                    self.assertTrue(point[dim] >= linear_dims[dim][0])
                    self.assertTrue(point[dim] <= linear_dims[dim][1])


    # Testing calculating leaf variances

    def test_make_full_leaf_var_list(self):
        d = 3
        lbda = 0.5
        n_points = 100
        n_labelled = 20
        temp_tree = Mondrian_Tree([[0,1]]*d)
        temp_tree.update_life_time(lbda, set_seed=100)

        labelled_indicies = range(n_labelled)
        labels = [random.random() for i in range(n_labelled)]
        data = []
        random.seed(1)
        for i in range(n_points):
            point = []
            for j in range(d):
                point.append(random.random())
            data.append(point)
        temp_tree.input_data(data, labelled_indicies, labels)
        temp_tree.make_full_leaf_list()
        temp_tree.make_full_leaf_var_list()

        for i, node in enumerate(temp_tree._full_leaf_list):
            node_labels = [temp_tree.labels[x] for x in node.labelled_index]
            # print(node_labels)
            if len(node_labels) != 0:
                temp_mean = sum(node_labels)/len(node_labels)
                # print(temp_mean)
                temp_var = 1/(len(node_labels)-1) * sum([(x-temp_mean)**2 for x in node_labels])
                self.assertTrue(abs(temp_tree._full_leaf_var_list[i] - temp_var) < 1e-9)
            else:
                self.assertEqual(temp_tree._full_leaf_var_list[i],0)

    # Testing calculating leaf mean

    def test_make_full_mean_list(self):
        d = 3
        lbda = 0.5
        n_points = 100
        n_labelled = 20
        temp_tree = Mondrian_Tree([[0,1]]*d)
        temp_tree.update_life_time(lbda, set_seed=100)

        labelled_indicies = range(n_labelled)
        labels = [random.random() for i in range(n_labelled)]
        data = []
        random.seed(1)
        for i in range(n_points):
            point = []
            for j in range(d):
                point.append(random.random())
            data.append(point)
        temp_tree.input_data(data, labelled_indicies, labels)
        temp_tree.make_full_leaf_list()
        temp_tree.make_full_leaf_mean_list()

        for i, node in enumerate(temp_tree._full_leaf_list):
            node_labels = [temp_tree.labels[x] for x in node.labelled_index]
            # print(node_labels)
            if len(node_labels) != 0:
                temp_mean = sum(node_labels)/len(node_labels)
                # print(temp_mean)
                self.assertTrue(abs(temp_tree._full_leaf_mean_list[i] - temp_mean) < 1e-9)
            else:
                self.assertEqual(temp_tree._full_leaf_mean_list[i],0)

    # Testing calculating leaf marginal probabilities

    def test_make_full_marginal_list(self):
        d = 3
        lbda = 0.5
        n_points = 100
        n_labelled = 20
        temp_tree = Mondrian_Tree([[0,1]]*d)
        temp_tree.update_life_time(lbda, set_seed=100)

        labelled_indicies = range(n_labelled)
        labels = [random.random() for i in range(n_labelled)]
        data = []
        random.seed(1)
        for i in range(n_points):
            point = []
            for j in range(d):
                point.append(random.random())
            data.append(point)
        temp_tree.input_data(data, labelled_indicies, labels)
        temp_tree.make_full_leaf_list()
        temp_tree.make_full_leaf_marginal_list()

        for i, node in enumerate(temp_tree._full_leaf_list):
            node_points = (
                [temp_tree.points[x] for x in node.labelled_index]+
                [temp_tree.points[x] for x in node.unlabelled_index])
            # print(node_labels)
            temp_marginal = len(node_points)/temp_tree._num_points
            self.assertTrue(abs(temp_tree._full_leaf_marginal_list[i] - temp_marginal) < 1e-9)

    # Testing using predictions

    def test_predict_bad_input(self):
        with self.assertRaises(TypeError):
            self.mt1.predict(1)

    def test_predict_bad_length(self):
        with self.assertRaises(ValueError):
            self.mt1.predict([])

    def test_predict_empty(self):
        with self.assertWarns(UserWarning):
            self.mt1.predict([0.5]*self.d)

    def test_predict_no_leaf_list(self):
        d = 3
        lbda = 0.5
        n_points = 100
        n_labelled = 20
        seed = 1
        temp_tree = Mondrian_Tree([[0,1]]*d)
        temp_tree.update_life_time(lbda, set_seed=100)

        labelled_indicies = range(n_labelled)
        labels = [random.random() for i in range(n_labelled)]
        data = []
        random.seed(seed)
        for i in range(n_points):
            point = []
            for j in range(d):
                point.append(random.random())
            data.append(point)
        temp_tree.input_data(data, labelled_indicies, labels)

        new_point = []
        for j in range(d):
            new_point.append(random.random())
        random.seed(seed)
        pred = temp_tree.predict(new_point)
        print(pred)

        node = temp_tree._root.leaf_for_point(new_point)
        vals = [labels[x] for x in node.labelled_index]
        print(len(vals))
        self.assertEqual(pred, sum(vals)/len(vals))

    def test_predict_leaf_list(self):
        d = 3
        lbda = 0.5
        n_points = 100
        n_labelled = 20
        seed = 1
        temp_tree = Mondrian_Tree([[0,1]]*d)
        temp_tree.update_life_time(lbda, set_seed=100)
        temp_tree.make_full_leaf_list()

        labelled_indicies = range(n_labelled)
        labels = [random.random() for i in range(n_labelled)]
        data = []
        random.seed(seed)
        for i in range(n_points):
            point = []
            for j in range(d):
                point.append(random.random())
            data.append(point)
        temp_tree.input_data(data, labelled_indicies, labels)

        new_point = []
        for j in range(d):
            new_point.append(random.random())
        random.seed(seed)
        pred = temp_tree.predict(new_point)
        print(pred)

        node = temp_tree._root.leaf_for_point(new_point)
        vals = [labels[x] for x in node.labelled_index]
        print(len(vals))
        self.assertEqual(pred, sum(vals)/len(vals))

    # Testing theoretical bounds

    def test_expected_split_bound(self):
        reps = 100
        tot = 0
        lbda = 5
        d = 3
        for i in range(reps):
            temp = Mondrian_Tree([[0,1]]*d)
            temp.update_life_time(lbda,set_seed=i)
            tot += temp._num_leaves - 1
        # print(tot/reps, ((1+lbda)*math.exp(1))**d)
        self.assertTrue(tot/reps< ((1+lbda)*math.exp(1))**d)


    def test_expected_cell_diameter_bounds(self):
        tot = 0
        lbda = 5
        self.mt1.update_life_time(lbda, set_seed=1)
        self.mt1.make_full_leaf_list()
        for node in self.mt1._full_leaf_list:
            tot += node.calculate_cell_l2_diameter()**2

        # print(tot/len(self.mt1._full_leaf_list), 4*self.d/lbda**2)
        self.assertTrue(tot/len(self.mt1._full_leaf_list) < 4*self.d/lbda**2)

    def test_probabilistic_cell_diameter_bounds(self):
        lbda = 5
        deltas = [0.1,0.2,0.3,0.4,0.5,1,1.5]
        self.mt1.update_life_time(lbda, set_seed=1)
        self.mt1.make_full_leaf_list()
        # print(len(self.mt1._full_leaf_list))
        for delta in deltas:
            tot = 0
            for node in self.mt1._full_leaf_list:
                tot += int(node.calculate_cell_l2_diameter() > delta)

            # print(tot/len(self.mt1._full_leaf_list), 
            #     self.d * (1+ (lbda * delta)/math.sqrt(self.d))*
            #     math.exp(-lbda * delta/math.sqrt(self.d)))
            self.assertTrue(tot/len(self.mt1._full_leaf_list) < 
                self.d * (1+ (lbda * delta)/math.sqrt(self.d))*
                math.exp(-lbda * delta/math.sqrt(self.d)))

    def test_update_life_time_tuned(self):

        self.mt1.update_life_time(self.a * self.n**(1/(2+self.d) - 1),set_seed=1)
        self.assertEqual(self.mt1._num_leaves,3)



if __name__ == '__main__':
    unittest.main()