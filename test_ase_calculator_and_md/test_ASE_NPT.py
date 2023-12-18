# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:13:07 2023

@author: ZHANG Jun
"""

import os
import json
from ase.io import read
from ase import units
from ase.calculators.calculator import Calculator
# from ase.calculators.calculator import all_changes

import torch
from agat.data.build_dataset import CrystalGraph
from agat.lib.model_lib import load_model

class AgatCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']
    default_parameters  = { }
    ignored_changes = set()
    def __init__(self, model_save_dir, graph_build_scheme_dir, device = 'cuda',
                 **kwargs):
        Calculator.__init__(self, **kwargs)

        self.model_save_dir = model_save_dir
        self.graph_build_scheme_dir = graph_build_scheme_dir
        self.device = device

        self.model = load_model(self.model_save_dir, self.device)
        self.graph_build_scheme = self.load_graph_build_scheme(self.graph_build_scheme_dir)

        build_properties = {'energy': False, 'forces': False, 'cell': False,
                            'cart_coords': False, 'frac_coords': False, 'path': False,
                            'stress': False} # We only need the topology connections.
        self.graph_build_scheme['build_properties'] = {**self.graph_build_scheme['build_properties'],
                                                       **build_properties}
        self.cg = CrystalGraph(**self.graph_build_scheme)

    def load_graph_build_scheme(self, path):
        """ Load graph building scheme. This file is normally saved when you build your dataset.

        :param path: Directory for storing ``graph_build_scheme.json`` file.
        :type path: str
        :return: A dict denotes how to build the graph.
        :rtype: dict

        """
        json_file  = os.path.join(path, 'graph_build_scheme.json')
        assert os.path.exists(json_file), f"{json_file} file dose not exist."
        with open(json_file, 'r') as jsonf:
            graph_build_scheme = json.load(jsonf)
        return graph_build_scheme

    def calculate(self, atoms=None, properties=None, system_changes=['positions', 'numbers', 'cell', 'pbc']):
        """

        :param atoms: ase.atoms object, defaults to None
        :type atoms: ase.atoms, optional
        :param properties: calculated properties, defaults to None
        :type properties: none, optional
        :param system_changes: DESCRIPTION, defaults to ['positions', 'numbers', 'cell', 'pbc']
        :type system_changes: TYPE, optional
        :return: calculated results
        :rtype: dict

        """

        if atoms is not None:
            self.atoms = atoms.copy()

        if properties is None:
            properties = self.implemented_properties

        # read graph
        graph, info = self.cg.get_graph(atoms)
        graph = graph.to(self.device)

        with torch.no_grad():
            energy_pred, force_pred, stress_pred = self.model.forward(graph)
        self.results = {'energy': energy_pred[0].item() * len(atoms),
                        'forces': force_pred.cpu().numpy(),
                        'stress': stress_pred[0].cpu().numpy()}

model_save_dir = os.path.join('acatal', 'test', 'agat_model')
graph_build_scheme_dir = model_save_dir
fname = os.path.join('acatal', 'test', 'POSCAR_NiCoFePdPt')

# NPT simulations (the isothermal-isobaric ensemble) Failed to optimized the cell shape.
from ase.md.npt import NPT
from ase.md import MDLogger
atoms = read(fname)
calculator = AgatCalculator(model_save_dir, graph_build_scheme_dir, device='cpu')
atoms.set_calculator(calculator)
dyn = NPT(atoms, timestep=5.0 * units.fs, temperature_K=800,
               ttime = 25 * units.fs,
               externalstress = [0.0] * 6,
               mask=None,
               trajectory=os.path.join('acatal', 'test', 'md_NPT.traj'))

dyn.attach(MDLogger(dyn, atoms, os.path.join('acatal', 'test', 'md_NPT.log'),
                    header=True,
                    stress=True,
                    peratom=True,
                    mode="w"), interval=1)
dyn.run(100)

# calculator results
atoms.get_total_energy()
''' Out:
-660.612602431774
'''

atoms.get_stress()
''' Out: Note that these are randomly small numbers.
array([-5.7220459e-06,  0.0000000e+00,  5.2452087e-06,  0.0000000e+00,
       -1.1444092e-05,  9.5367432e-07], dtype=float32)
'''

atoms.get_forces()
''' Out:
array([[ 0.6747068 ,  0.4705114 ,  0.35780022],
       [ 0.7946833 ,  0.1836735 , -0.26059636],
       [-0.5386316 ,  0.33120936,  0.42413437],
       [-0.69506496, -0.44159696, -1.409261  ],
       [-1.3713082 ,  1.2892114 , -0.24743134],
       [ 0.81071734, -0.31286642,  0.177412  ],
       [ 2.9615865 , -1.8013232 , -0.58083194],
       [-2.7494783 ,  0.51401556,  0.49028185],
       [-0.5820948 ,  0.797262  , -1.6618928 ],
       [-0.21648164, -1.6717458 ,  1.2185351 ],
       [-1.3973591 ,  0.20589188, -1.377053  ],
       [ 0.69369715,  0.61521137,  0.50114924],
       [ 0.10086954,  1.2272459 ,  0.00698882],
       [-0.15045267,  0.11749014,  0.6022616 ],
       [ 1.1790551 ,  0.63241893,  0.15979218],
       [-0.14504828, -2.0510945 ,  0.3182725 ],
       [-1.8011019 ,  2.0863445 , -0.33628273],
       [ 0.5205651 , -0.7771849 ,  0.6096837 ],
       [ 1.256319  , -0.9233563 ,  0.31635296],
       [ 1.3010176 , -1.0837101 , -0.29183865],
       [ 0.8155178 ,  2.4915977 ,  1.4422597 ],
       [-0.19606555,  0.6907968 , -0.4278219 ],
       [ 2.443568  ,  0.41107315, -0.42305875],
       [ 0.6203102 , -0.17084925, -0.36178878],
       [ 0.0257947 , -0.8117402 ,  0.3685431 ],
       [ 0.25919944, -0.04894231, -1.2017149 ],
       [-0.10250738,  1.2482839 ,  2.3008206 ],
       [ 0.2703502 ,  0.3071812 ,  0.25912374],
       [ 0.7575261 ,  0.68934685,  1.0650421 ],
       [-1.3871952 , -1.5511563 ,  1.6791598 ],
       [ 0.963699  ,  0.20295277,  0.6373118 ],
       [-0.930995  ,  0.58101946,  0.12398439],
       [ 1.6814932 , -2.02139   , -2.1228275 ],
       [-0.3508272 ,  0.97826296,  0.6304993 ],
       [-2.7615857 , -0.92907715,  0.26817456],
       [ 0.3434448 , -1.3601404 ,  1.662419  ],
       [-1.2042409 , -1.1254203 , -1.6011688 ],
       [-0.8494062 , -0.962777  , -0.8946191 ],
       [ 0.46963727, -0.19446968,  0.3270494 ],
       [-1.5233426 , -1.2909007 ,  1.229757  ],
       [ 2.2071888 ,  2.0791895 ,  0.91254306],
       [-0.5897356 ,  0.898044  , -1.0278081 ],
       [-0.00706768,  0.4835253 ,  0.41542128],
       [ 0.6589559 , -1.4192609 , -0.36512226],
       [ 0.93414587,  1.3234407 ,  0.5950692 ],
       [-0.84704816, -0.04837503, -0.36872077],
       [-0.42867014, -0.59150076, -0.50996053],
       [-0.41680804,  1.0352253 , -1.1710743 ],
       [ 1.5981714 ,  0.64289075,  1.2290689 ],
       [ 0.09698351, -0.41874605,  0.5088653 ],
       [-0.15575172, -0.66580886,  0.05152826],
       [-0.98578715,  1.8644161 , -2.510659  ],
       [ 0.14627706, -0.09590377,  0.8611695 ],
       [ 0.05031968,  0.24151395,  0.3233752 ],
       [ 0.9200138 ,  0.981933  ,  2.9019601 ],
       [-0.99986476, -0.18329075, -0.8793778 ],
       [-1.4979514 , -0.0595777 ,  1.6699375 ],
       [ 0.95175165, -0.7075956 ,  0.83289963],
       [-0.7164982 , -1.2978745 , -1.1598405 ],
       [ 1.2101483 ,  1.2167443 , -0.21190752],
       [-0.5609931 ,  0.31482825, -0.2986711 ],
       [-1.7392095 , -1.0167887 , -0.12907603],
       [ 0.16959836,  0.14469211,  1.2221037 ],
       [ 0.9064245 , -0.07791211, -1.5685668 ],
       [ 1.2553368 , -0.46813905, -1.3581388 ],
       [ 0.91802824,  0.25224847, -1.5425776 ],
       [-0.14212547,  0.05432639,  0.6333222 ],
       [ 0.4954604 , -1.0745939 , -0.23032707],
       [-0.613876  ,  0.08119614,  1.0194129 ],
       [-0.54598683, -0.9208951 ,  0.1902876 ],
       [ 0.77080095, -0.17882392, -0.18709782],
       [ 1.019786  , -0.28463954,  0.5891782 ],
       [ 1.1363664 , -1.6811172 , -0.35681763],
       [-0.1433167 , -0.3172658 , -0.42904788],
       [-1.1976106 , -0.7454058 , -0.44476277],
       [-0.7140731 ,  0.49790865, -0.56762844],
       [-1.4297545 ,  0.6405293 , -0.15905748],
       [ 0.36893687,  1.0990553 , -0.2742057 ],
       [-0.49121922, -2.1957603 , -1.5569761 ],
       [ 0.0311913 ,  0.32106858,  1.433052  ],
       [ 0.38478255,  0.8560324 ,  0.30599287],
       [ 0.15306172, -0.40413702, -0.07003991],
       [ 0.60496384,  2.521227  ,  0.1988086 ],
       [-0.23362994, -0.19003665, -0.65220994],
       [-0.44183582, -0.54187566, -2.0261953 ],
       [-0.6733656 ,  1.6372622 ,  0.8223839 ],
       [-0.6587424 , -0.7543673 , -1.6062207 ],
       [-1.1040184 , -0.44418314,  0.3703324 ],
       [-1.4919062 ,  0.0258675 , -0.31818926],
       [-0.05402571,  0.13014673,  1.268218  ],
       [ 0.15603876, -0.10840611, -0.07443869],
       [ 1.0393914 ,  0.77921563,  0.21658783],
       [ 1.5402045 ,  0.24393108,  0.00595617],
       [ 0.11211825,  0.19176844,  0.31934622],
       [-0.22894335,  0.22553268, -0.2898022 ],
       [-0.71720123, -0.43274015, -0.5309233 ]], dtype=float32)
'''

# cell before simulation
atoms = read(fname)
atoms.cell.array
''' Out:
array([[10.42002806,  0.        ,  0.        ],
       [ 0.        ,  9.024009  ,  0.        ],
       [ 0.        ,  0.        , 12.76187592]])
'''

# cell after simulation
''' Out:
array([[10.42002806,  0.        ,  0.        ],
       [ 0.        ,  9.024009  ,  0.        ],
       [ 0.        ,  0.        , 12.76187592]])
'''
