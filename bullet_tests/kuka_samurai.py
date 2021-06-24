import math
import time

import pybullet as p
import pybullet_data

EARTHS_AVG_GRAVITY = 9.80665

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -EARTHS_AVG_GRAVITY)

p.loadURDF('plane.urdf')

kuka_id, = p.loadSDF('kuka_iiwa/kuka_with_gripper.sdf')
print(kuka_id)
num_joints = p.getNumJoints(kuka_id)
print(num_joints)
print(num)

# p.loadURDF("tray/tray.urdf", [0, 0, 0])
# p.loadURDF("block.urdf", [0, 0, 2])

# p.loadURDF('duck_vhacd.urdf', [0.7, -0.7, .1])
# p.loadURDF('lego/lego.urdf', [0.6, -0.6, .1])
# p.loadURDF('jenga/jenga.urdf', [0.5, -0.5, .1])
# p.loadURDF('sphere2.urdf', [1, 1, 1])
# p.loadURDF('sphere2red.urdf', [-1, -1, 1])
# p.loadURDF('teddy_vhacd.urdf', [.5, -.6, 0],
#            p.getQuaternionFromEuler([2, 0, math.tau * 5 / 8]),  globalScaling=4)
# p.loadURDF('block.urdf', [.7, -.3, .1])
# p.loadURDF('samurai.urdf')

for i in range(10_000):
  p.stepSimulation()
  time.sleep(1 / 240)

p.disconnect()
