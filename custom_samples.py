import numpy as np
from barcgp.common.pytypes import *
from barcgp.common.utils.scenario_utils import SampleGenerator, Sample
import pickle
import os
# Load X, Y from the dataset
X_data = []
Y_data = []

folder_path = '/home/jeanho/acados/race_cars/gp_data_batches/'
samples = []

for file_name in os.listdir(folder_path):
    full_path = os.path.join(folder_path, file_name)
    if os.path.isfile(full_path) and file_name.endswith('.npz'):
        data = np.load(full_path)
        X_data.append(data['X'])
        Y_data.append(data['Y'])
X_data = np.concatenate(X_data, axis=0)
Y_data = np.concatenate(Y_data, axis=0) 
print(X_data.shape, Y_data.shape)
print(type(X_data), type(Y_data))

for X, Y in zip(X_data, Y_data):
    
    ego_state = VehicleState(
        t=X[0],
        x = Position(x=X[1], y=Y[2], z=0),
        v = BodyLinearVelocity(v_long=X[5], v_tran =X[7], v_n=0),
        w = BodyAngularVelocity(w_psi=X[9]),
        a = BodyLinearAcceleration(a_long=X[11], a_tran=X[13], a_n=0),
        aa = BodyAngularAcceleration(), #기본값으로 0으로로
        e = OrientationEuler(psi=X[-13]),
        p=ParametricPose(s=X[-11], x_tran=X[-9], e_psi=X[-7]),
        pt = ParametricVelocity(), #기본값 0으로
        u = VehicleActuation(u_a=X[-5] ,u_steer=X[-3]),
        lookahead=TrackLookahead(curvature=('d', [X[-1], 0, 0])),
        v_x=0, v_y=0, lap_num=None
    )

    tar_state = VehicleState(
        t=X[0],
        x = Position(x=X[3], y=Y[4], z=0),
        v = BodyLinearVelocity(v_long=X[6], v_tran =X[8] , v_n =0),
        w = BodyAngularVelocity(w_psi=X[10]),
        a = BodyLinearAcceleration(a_long=X[12], a_tran=[14], a_n=0), 
        aa = BodyAngularAcceleration(), #기본값으로 0으로로
        e = OrientationEuler(psi=X[-12]),
        p = ParametricPose(s=X[-10], x_tran=X[-8], e_psi=X[-6]),
        pt = ParametricVelocity(), #기본값 0으로
        u = VehicleActuation(u_a=X[-4] ,u_steer=X[-2]),
        lookahead=TrackLookahead(curvature=('d', [X[-1], 0, 0])),
        v_x=0, v_y=0, lap_num=None
    )

    # Reconstruct delta TV state (output)
    dtar_state = VehicleState(
        t=X[0],
        x=Position(x=X[3], y=Y[4], z=0),
        v=BodyLinearVelocity(v_long=Y[-3], v_tran=Y[-2]),
        w=BodyAngularVelocity(w_psi=Y[-1]),
        a=BodyLinearAcceleration(a_long=X[12], a_tran=[14], a_n=0),
        aa=BodyAngularAcceleration(), #기본값으로 0으로로
        e=OrientationEuler(psi=X[-12]),
        p=ParametricPose(s=Y[0], x_tran=Y[1], e_psi=Y[2]),
        pt=ParametricVelocity(), #기본값 0으로
        u=VehicleActuation(u_a=X[-4] ,u_steer=X[-2]),
        lookahead=TrackLookahead(curvature=('d', [X[-1], 0, 0])),
        v_x=0, v_y=0, lap_num=None
    )


    # Create a Sample object
    sample = Sample(input=(ego_state, tar_state), output=dtar_state, s=X[-1])
    samples.append(sample)

with open("sample_batch", "wb") as f:
    pickle.dump(samples, f)

print(f"Loaded {len(samples)} samples!")
