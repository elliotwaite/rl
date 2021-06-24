import time

import pybullet as p


p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0, 0, -10)
planeId = p.loadURDF('plane.urdf')
cubeStartPos = [0, 0, 1]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
# kuka_iiwa_model_id = p.loadURDF('kuka_iiwa/model.urdf', [0, 0, 0], useFixedBase=True)
# kuka_iiwa_model_id = p.loadSDF('kuka_iiwa/kuka_with_gripper.sdf')
# kuka_iiwa_model_id = p.loadSDF('kuka_iiwa/kuka_with_gripper2.sdf')
# kuka_iiwa_model_id = p.loadSDF('kuka_iiwa/model.sdf')
# kuka_iiwa_model_id = p.loadURDF('kuka_iiwa/model.urdf', [0, 0, 0], useFixedBase=True)
# kuka_iiwa_model_id = p.loadURDF('kuka_iiwa/model_free_base.urdf', [0, 0, 0], useFixedBase=True)
kuka_iiwa_model_id = p.loadURDF('kuka_iiwa/model_vr_limits.urdf', [0, 0, 0], useFixedBase=True)
# boxId = p.loadURDF('r2d2.urdf', cubeStartPos, cubeStartOrientation)
for i in range(10000):
    p.stepSimulation()
    time.sleep(1/240)
# cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
# print(cubePos, cubeOrn)
p.disconnect()
