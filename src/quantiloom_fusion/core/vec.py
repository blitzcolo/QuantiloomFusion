"""Pure-Python vector and pose math (right/down/forward camera convention)."""

import math


def add(a, b): return [x+y for x,y in zip(a,b)]
def scale(a, s): return [x*s for x in a]
def dot(a, b): return sum(x*y for x,y in zip(a,b))
def cross(a, b): return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def norm(a): return math.sqrt(dot(a,a))
def unit(a):
    n=norm(a)
    if n<1e-10: raise ValueError("degenerate camera basis")
    return scale(a,1/n)


def pose(position, target, up):
    forward=unit(add(target,scale(position,-1)))
    right=unit(cross(forward,up))
    down=unit(cross(forward,right))
    return [right[0],down[0],forward[0],position[0],
            right[1],down[1],forward[1],position[1],
            right[2],down[2],forward[2],position[2],0,0,0,1]


def local_rotation(yaw, pitch, roll):
    # Right/down/forward convention; yaw about down, pitch about right, roll about forward.
    y,p,r=map(math.radians,(yaw,pitch,roll))
    ry=[[math.cos(y),0,math.sin(y)],[0,1,0],[-math.sin(y),0,math.cos(y)]]
    rx=[[1,0,0],[0,math.cos(p),-math.sin(p)],[0,math.sin(p),math.cos(p)]]
    rz=[[math.cos(r),-math.sin(r),0],[math.sin(r),math.cos(r),0],[0,0,1]]
    def mul(a,b):return [[sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    return mul(mul(ry,rx),rz)


def camera_pose(rig_pose, camera, world_scale):
    R=[[rig_pose[i*4+j] for j in range(3)] for i in range(3)]
    offset=scale(camera["offset_rdf_m"],1/world_scale)
    origin=[rig_pose[i*4+3]+dot(R[i],offset) for i in range(3)]
    local=local_rotation(*camera["rotation_ypr_deg"])
    axes=[[sum(R[i][k]*local[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    return [*axes[0],origin[0],*axes[1],origin[1],*axes[2],origin[2],0,0,0,1]
