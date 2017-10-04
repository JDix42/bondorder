import numpy as np
import matplotlib.pyplot as plt
import gra_sheet_gen as gsg
import math
from numpy.testing import *
from bopcalcpl import psi4_calc, preprocess, calc_angles, psin_calc, bopcalc, replicate_1D


class TestPsi4(TestCase):
    def setUp(self):

        self.pl = []
        
        self.p4 = psi4_calc
        
        array = np.zeros((100,3))
        for i in range(10):
            array[i*10:(1+i)*10,0] = i
            for j in range(10):
                array[j + i*10,1] = j
                self.pl.append(1)

        self.points = np.zeros((np.shape(array)))
        self.points[:,0] = array[:,0]
        self.points[:,1] = array[:,1]
        self.points[:,2] = array[:,2]

        max_p = np.max(self.points[:,:], axis=0)
        min_p = np.min(self.points[:,:], axis=0)
        self.width = max_p[:] - min_p[:]
        self.mv = np.array([min_p[0] - 1, max_p[0], min_p[1] -1 , max_p[1], min_p[2], max_p[2]])
        self.md = 5
        self.sv = [4,6]
        self.bv = np.zeros(( len(self.sv), 2))

        self.max_dist = 0.5
        self.sens_test = 5
        self.v4 = 0
        self.v6 = 0
        
        self.var = []
        for p in range(10):
            for i in range(1,10):
                val = i * math.pow(10, -10+p)
                self.var.append(val)

        #print self.var

        self.tp = np.shape(self.points)[0]

        bondorder_values = []

        self.symval = [4]

        self.p_x = replicate_1D(self.points, max_p[0], min_p[0], 0, abs(max_p[0] - min_p[0]), 0)
        self.p_xy = replicate_1D(self.p_x, max_p[1], min_p[1], 0, abs(max_p[1] - min_p[1]), 1)


        self.a1, self.a2, self.radfp, self.bv, self.ridge_points, self.dist = preprocess(
            self.points, self.tp, self.symval, bondorder_values, [0,1], 5)

        self.rcan4, self.ccan4, self.rca4, self.cca4 = calc_angles(
            4, self.radfp, self.a1, self.a2)

        self.rcan6, self.ccan6, self.rca6, self.cca6 = calc_angles(
            6, self.radfp, self.a1, self.a2)

        self.bo4, rp, cp, bo4, bo4err = psi4_calc(self.tp, self.ridge_points, self.a1, self.rca4, self.cca4, self.dist)

        self.bo6, rp, cp, bo6, bo6err = psin_calc(
            6, self.radfp, self.a1, self.a2, self.tp, self.ridge_points, self.dist, self.points) 

        #shift = np.zeros((np.shape(self.points)))
        #shift[:,2] = self.points[:,2] + self.width[2] + 1  
        #shift[:,1] = self.points[:,1]
        #shift[:,0] = self.points[:,0]
        #self.ppl = shift

        #shift = np.zeros((np.shape(self.points)))
        #shift[:,2] = self.points[:,2] - self.width[2] -1 
        #shift[:,1] = self.points[:,1]
        #shift[:,0] = self.points[:,0]
        #self.pmn = shift

        #self.bf = np.concatenate([self.ppl, self.pmn], axis=0)

        print np.shape(self.bv), "bv", len(self.sv)

        self.bv = np.zeros((len(self.sv),2))
             
        self.bo = bopcalc(self.points,  self.sv, self.bv, self.mv, self.md)
        
    def tearDown(self):
        del self.p4
        del self.points
        del self.a1, self.a2,
        del self.symval
        del self.radfp, self.bv, self.ridge_points
        del self.dist
        del self.rcan4, self.ccan4, self.rca4, self.cca4
        del self.rcan6, self.ccan6, self.rca6, self.cca6

    def test_psi4_sq(self):     
        assert_almost_equal(self.bo4, 1, err_msg="Psi 4 doesn't quite work right")
        assert_almost_equal(self.bo[0,0], 1, err_msg="psi4 doesn't work right")
        assert_almost_equal(self.bo[0,0], self.bo4, err_msg="Both psi4 values don't aggree")

        if self.bo4<=1:
            truth=True
        else:
            truth=False
        
        assert_equal(truth, True, err_msg="Psi 4 is not normalised correctly")

    def test_psi6_sq(self):     
        assert_almost_equal(self.bo6, 0, err_msg="Psi 6 with square lattice doesn't quite work")
        assert_almost_equal(self.bo[1,0], 0, err_msg="Psi 6 with square lattice doesn't quite work")
        assert_almost_equal(self.bo[1,0], self.bo6, err_msg="both psi6 values don't agree")

    def test_scale_sq(self):
        test_no = 50
        scale_list = np.random.rand((test_no))
        scale_list[0] = 1

        for t in range(test_no):            
            scale = scale_list[t]

            r = self.points[:,:]*scale
            sht_p = np.zeros((np.shape(r)))
            sht_p[:,2] = r[:,2] #+ scale * ( self.width[2] + 1)
            sht_p[:,1] = r[:,1] + scale * ( self.width[2] + 1)
            sht_p[:,0] = r[:,0]

            sht_n = np.zeros((np.shape(r)))
            sht_n[:,2] = r[:,2] #- scale * ( self.width[2] + 1)
            sht_n[:,1] = r[:,1] - scale * ( self.width[2] + 1)
            sht_n[:,0] = r[:,0]
          
            #bf = np.concatenate([sht_p, sht_n], axis=0)

            mv = self.mv * scale
            md = self.md * scale
            
            bo = bopcalc(r[:,:],  self.sv, self.bv, mv, md)
            
            psi4 = bo[0,0] / self.bo[0,0]
            psi6 = bo[1,0] / self.bo[1,0]

            diff4 = bo[0,0] - self.bo[0,0]
            diff6 = bo[1,0] - self.bo[1,0]

            error4 = np.abs(diff4)/self.bo[0]
            error6 = np.abs(diff6)/self.bo[1]

            print psi4, psi6, diff4, diff6, scale
            print error4, error6

            assert_array_less(error4, 0.000001, err_msg="psi4 on square error on scaling is too large")
            assert_array_less(error6, 0.000001, err_msg="psi6 on sqaure error on scaling is too large")
    
    def _sensitive_sq4(self,d):
        for v in range(len(self.var)):
            for t in range(self.sens_test):                        
                ran = np.random.rand(np.shape(self.points)[0], np.shape(self.points)[1])
                ran *= (self.var[v] * self.max_dist / 100)
                r_points = self.points[:,:] + ran[:,:]

                bo = bopcalc(r_points, self.sv, self.bv, self.mv, self.md)        

                assert_almost_equal(bo[0,0], self.bo[0,0], decimal = d, err_msg="psi 4 stops working for " + str(self.v4))
            
            print "Percentange of max_dist is ", self.v4
            print "Variance in Displacment is", self.v6 * self.max_dist
            print "Precision used was to the " + str(d) + " decimal place"
            self.v4=self.var[v]

    def _sensitive_sq6(self,d):
        for v in range(len(self.var)):
            for t in range(self.sens_test):                        
                ran = np.random.rand(np.shape(self.points)[0], np.shape(self.points)[1])
                ran *= (self.var[v] * self.max_dist / 100)
                r_points = self.points[:,:] + ran[:,:]

                bo = bopcalc(r_points, self.sv, self.bv, self.mv, self.md)        

                assert_almost_equal(bo[1,0], self.bo[1,0], decimal = d, err_msg="psi 6 stops working for " + str(self.v6))

            print "Percentange of max_dist is ", self.v6
            print "Variance in Displacment is", self.v6 * self.max_dist
            print "Precision used was to the " + str(d) + " decimal place"
            self.v6=self.var[v]

    #def test_sens_sq4_7(self):
    #    self._sensitive_sq4(7)

    #def test_sens_sq4_6(self):
    #    self._sensitive_sq4(6)
        
    #def test_sens_sq4_5(self):
    #    self._sensitive_sq4(5)

    #def test_sens_sq4_4(self):
    #    self._sensitive_sq4(4)
        
    #def test_sens_sq4_3(self):
    #    self._sensitive_sq4(3)

    #def test_sens_sq4_2(self):
    #    self._sensitive_sq4(2)

    #def test_sens_sq4_1(self):
    #    self._sensitive_sq4(1)

    #def test_sens_sq4_0(self):
    #    self._sensitive_sq4(0)

    #def test_sens_sq6_7(self):
    #    self._sensitive_sq6(7)

    #def test_sens_sq6_6(self):
    #    self._sensitive_sq6(6)
        
    #def test_sens_sq6_5(self):
    #    self._sensitive_sq6(5)

    #def test_sens_sq6_4(self):
    #    self._sensitive_sq6(4)
        
    #def test_sens_sq6_3(self):
    #    self._sensitive_sq6(3)

    #def test_sens_sq6_2(self):
    #    self._sensitive_sq6(2)

    #def test_sens_sq6_1(self):
    #    self._sensitive_sq6(1)

    #def test_sens_sq6_0(self):
    #    self._sensitive_sq6(0)

class TestLarge(TestCase):
    def setUp(self):
        points = 1000
        self.p = np.random.rand(points, 3)
        
        shift = np.zeros((np.shape(self.p)))
        shift[:,2] = self.p[:,2] 
        shift[:,1] = self.p[:,1]
        shift[:,0] = self.p[:,0]
        self.ppl = shift

        shift = np.zeros((np.shape(self.p)))
        shift[:,2] = self.p[:,2]  
        shift[:,1] = self.p[:,1]
        shift[:,0] = self.p[:,0]
        self.pmn = shift

        #self.bf = np.concatenate([self.ppl, self.pmn], axis=0)

        self.sv = [4,6]
        self.bv = np.zeros(( len(self.sv), 2))        
        self.mv = np.array([0,1,0,1,0,1])

        self.md = 1
        
        self.test_no = 50

        self.scale = np.random.rand(self.test_no)
        self.scale[0] = 1

        self.test_rot =6
        self.rot = np.random.rand(self.test_no+5)
        self.rot[0] = 0
        self.rot[1] = 0.5
        self.rot[2] = 0.25
        self.rot[3] = 0.75
        self.rot[4] = 1
        self.rot[5] = 0.1
        self.rot *= 2*math.pi
             
        self.bo = bopcalc(self.p,  self.sv, self.bv, self.mv, self.md)

    def test_psi4(self):
        if self.bo[0,0]<=1 and self.bo[0,0]>0:
            t1 = True
            t2 = True
        elif self.bo[0,0]<=1 and self.bo[0,0]<0:
            t1 = True
            t2 = False
        else:
            t1 = False
            t2 = True

        assert_equal(t1, True, err_msg="Psi 4 isn't normalised correctly")
        assert_equal(t2, True, err_msg="Psi 4 is less than zero?!")

    def test_psi6(self):
        if self.bo[1,0]<=1 and self.bo[1,0]>0:
            t1 = True
            t2 = True
        elif self.bo[1,0]<=1 and self.bo[1,0]<0:
            t1 = True
            t2 = False
        else:
            t1 = False
            t2 = True

        assert_equal(t1, True, err_msg="Psi 6 isn't normalised correctly")
        assert_equal(t2, True, err_msg="Psi 6 is less than zero?!")

    def test_scale(self):        
        for t in range(self.test_no):            
            scale = self.scale[t]

            r = self.p[:,:]*scale
            sht_p = np.zeros((np.shape(r)))
            sht_p[:,2] = r[:,2] 
            sht_p[:,1] = r[:,1] + scale
            sht_p[:,0] = r[:,0]

            sht_n = np.zeros((np.shape(r)))
            sht_n[:,2] = r[:,2] 
            sht_n[:,1] = r[:,1] - scale
            sht_n[:,0] = r[:,0]
          
            #bf = np.concatenate([sht_p, sht_n], axis=0)

            mv = self.mv * scale
            md = self.md * scale
            
            bo = bopcalc(r[:,:],  self.sv, self.bv, mv, md)
            
            #psi4 = bo[0] / self.bo[0]
            #psi6 = bo[1] / self.bo[1]

            diff4 = bo[0,0] - self.bo[0,0]
            diff6 = bo[1,0] - self.bo[1,0]

            error4 = np.abs(diff4)/self.bo[0,0]
            error6 = np.abs(diff6)/self.bo[1,0]

            #print psi4, psi6, diff4, diff6, scale
            print error4, error6

            assert_array_less(error4, 0.03, err_msg="psi4 error on scaling is too large")
            assert_array_less(error6, 0.000001, err_msg="psi6 error on scaling is too large")

    def test_rot(self):
        for rt in range(self.test_rot+5):            
            sint = math.sin(self.rot[rt])
            cost = math.cos(self.rot[rt])

            r_temp = self.p

            #print r_temp
            
            #print sint, cost
            if rt>5:                
                dim = [0,1]
                box_max = [1,1,1]
                box_min = [0,0,0]

                for d in range(len(dim)):
                    width = box_max[dim[d]] - box_min[dim[d]]
            
                    r_psh = np.zeros((np.shape(r_temp)))
                    r_psh[:,:] = r_temp[:,:]
                    r_psh[:,dim[d]] += width
            
                    r_msh = np.zeros((np.shape(r_temp)))
                    r_msh[:,:] = r_temp[:,:]
                    r_msh[:, dim[d]] -= width

                    r_all = np.concatenate([r_temp, r_psh, r_msh], axis=0)
                    r_temp = r_all

            q=r_temp
            r = np.zeros((np.shape(q)))

            #print r_temp

            r[:,0] = ((q[:,0] - 0.5) * cost + (q[:,1] - 0.5) * sint) + 0.5
            r[:,1] = (-(q[:,0] - 0.5) * sint + (q[:,1] - 0.5) * cost) + 0.5
            r[:,2] = q[:,1]  

            #print r
            
            p_box = np.where((r[:,0] < 1) & (r[:,0] > 0) & (r[:,1] < 1) & (r[:,1] > 0))

            #print np.shape(p_box)
            
            r_cut = r[p_box[0][:],:]

            sht_p = np.zeros(np.shape(r_cut))
            sht_p[:,1] = r_cut[:,1] + 1
            sht_p[:,2] = r_cut[:,2]
            sht_p[:,0] = r_cut[:,0]

            sht_n = np.zeros(np.shape(r_cut))
            sht_n[:,1] = r_cut[:,1] - 1        
            sht_n[:,2] = r_cut[:,2]
            sht_n[:,0] = r_cut[:,0]
 
            #bf = np.concatenate([sht_p, sht_n], axis=0)

            bo = bopcalc(r_cut[:,:], self.sv, self.bv, self.mv, self.md)
            
            psi4 = bo[0,0] / self.bo[0,0]
            psi6 = bo[1,0] / self.bo[1,0]

            diff4 = bo[0,0] - self.bo[0,0]
            diff6 = bo[1,0] - self.bo[1,0]

            error4 = np.abs(diff4)/self.bo[0,0]
            error6 = np.abs(diff6)/self.bo[1,0]

            print "test", psi4, psi6, diff4, diff6, error4, error6
            print "bo", bo[0,0], self.bo[0,0], bo[1,0], self.bo[1,0], bo[0,1], self.bo[0,1], bo[1,1], self.bo[1,1]
 
            #assert_almost_equal(psi4, 1, decimal=0, err_msg="psi4 doesn't work with coord rotation - ratio")
            #assert_almost_equal(psi6, 1, err_msg="psi6,doesn't work with coord rotation - ratio")        

            #assert_almost_equal(diff4, 0, decimal=3, err_msg="psi4 doesn't work with coord rotation - diff")
            #assert_almost_equal(diff6, 0, err_msg="psi6 doesn't work with coord rotation - diff")

            assert_array_less(error4, 0.00001, err_msg="psi4 error is too large with rotation")
            assert_array_less(error6, 0.00001, err_msg="psi6 error is too large with rotation")
            
class TestPsi6(TestCase):
    def setUp(self):
        self.org = gsg.graphene_create()
        n = 20
        m = 20
        
        self.g = gsg.replicate_graphene(self.org, n, m)

        self.gra=np.zeros((np.shape(self.g)))
        self.gra[:,0] = self.g[:,0]
        self.gra[:,1] = self.g[:,1]
        self.gra[:,2] = self.g[:,2]

        self.maxg = np.max(self.gra[:,:], axis=0)
        self.ming = np.min(self.gra[:,:], axis=0)
        
        self.width = self.maxg[:] - self.ming[:]

        shift = np.zeros(np.shape(self.gra))
        shift[:,2] = self.gra[:,2] + self.width[2] + 1.42
        shift[:,1] = self.gra[:,1]
        shift[:,0] = self.gra[:,0]
        sft_p = shift[:,:]

        shift = np.zeros(np.shape(self.gra))
        shift[:,2] = self.gra[:,2] - self.width[2] - 1.42
        shift[:,1] = self.gra[:,1]
        shift[:,0] = self.gra[:,0]
        sft_m = shift[:,:]

        #self.bf = np.concatenate([sft_p, sft_m], axis=0)
        
        self.sv = [4,6]
        self.bv = np.zeros((len(self.sv), 2))
        ring_shift = 1.42 * np.sin(30 * math.pi/180)
        #self.mv = np.array([self.maxg[2] + 0.71,self.ming[2] - 0.71,self.width[0] + ring_shift,self.maxg[0],self.ming[0]])
        self.mv = np.array([self.ming[0] - 1.42, self.maxg[0], self.ming[1]-1.42, self.maxg[1], self.ming[2], self.maxg[2]])

        self.md = 1.6

        gra_rand = np.random.rand(np.shape(self.gra)[0], np.shape(self.gra)[1]) * 10**-(7.5)
        self.gra += gra_rand

        #bf_rand = np.random.rand(np.shape(self.bf)[0], np.shape(self.gra)[1]) * 10**-(7.5)
        #self.bf += bf_rand

        self.bo = bopcalc(self.gra, self.sv, self.bv, self.mv, self.md)
        
    def test_psi4_hex(self):
        assert_almost_equal(self.bo[0,0], 0, decimal=1 , err_msg="psi4 isn't zero for a hexagonal lattice")
        
    def test_psi6_hex(self):
        assert_almost_equal(self.bo[1,0], 1, decimal=1, err_msg="psi6 isn't one for a hexagonal lattice")

        if self.bo[1,0] <= 1:
            truth = True
        else:
            truth = False
        
        assert_equal(truth, True, err_msg="Psi 6 is not normalised correctly")
        
