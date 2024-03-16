from utils import eucl_dist, dotProd, degToRad
import numpy as np
import math
import random

"""
this file contains the implementation of the search classes used in RRT
此文件包含 RRT 中使用的搜索类的实现
导入了2个文件：utils.py，environment.py
"""

"""
SearchNode
定义一些节点参数
"""

class SearchNode(object):
    def __init__(self,
                 state,
                 parent_node = None,
                 cost        = 0.0,
                 u           = (0, 0),
                 theta       = 0.0,
                 velocity    = (0.5, 0.0, 0.0),
                 length      = 1,
                 mass        = 0.5,
                 inertia     = 0.01,
                 ):

        self._parent    = parent_node   # pointing to parent SearchNode object  指向SearchNode的父对象
        self._state     = state   # Standard version: tuple (x,y)  就是(x,y)坐标
        self._u         = u     # 输入电压u
        self._cost      = cost   # 起始节点到该节点的总代价
        self._theta     = theta   # Positive anti-clockwise, wrt. x-axis 正逆时针
        self._velocity  = velocity
        self._length    = length
        self._m         = mass
        self._inertia   = inertia

    @property   # 装饰器property将state转换为只读属性，可以直接访问，不需要使用方法调用
    def state(self):
        return self._state

    @state.setter   # 用于定义state属性的setter方法,可以对state进行赋值
    def state(self, value):
        self._state = value  # 可以对state赋值

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        self._parent = value

    @property
    def cost(self):
        return self._cost

    @cost.setter
    def cost(self, value):
        self._cost = value

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self,value):
        self._u = value

    @property
    def theta(self):    # theta值固定
        return self._theta

    @property
    def velocity(self):
        return self._velocity

    @property
    def length(self):
        return self._length

    @property
    def m(self):
        return self._m

    @property
    def inertia(self):
        return self._inertia

    # 下面是定义了一些特殊方法
    def __repr__(self):   # 用于返回对象的字符串表示形式
        return "<SN (id: %s), (%0.2f,%0.2f), %0.2f, parent: %s>" % (id(self), self.state[0], self.state[1], self.cost, id(self.parent))

    def __eq__(self, other):   # 用于定义对象之间的相等性比较操作
        return isinstance(other, SearchNode) and self._state == other._state     # 基于state

    def __hash__(self):    # 定义对象的哈希值
        return hash(self._state)

    def __gt__(self, other):    # 定义对象之间的大于比较操作
        return self._cost > other._cost     # 基于cost


"""
Path class
路径类
"""

class Path(object):
    """
    This class computes the path from the starting state until the state specified by the search_node parameter by iterating backwards.
    该类通过向后迭代，计算从起始状态到 search_node 参数指定的状态的路径。
    """

    def __init__(self, search_node):
        self.path = []
        self.thetas = []
        self.inputs = []
        node = search_node
        while node is not None:
            self.path.append(node.state)
            self.thetas.append(node.theta)
            self.inputs.append(node.u)      # 输入电压u
            node = node.parent  # 反向迭代 遍历搜索节点的父节点

        self.path.reverse()   # 将path列表反转
        self.cost = search_node.cost
        # not correct if RRT* has rewired   如果 RRT*已重新布线，则不正确
        self.cost = 0   # 初始化cost为0
        for i in range(len(self.path)-1):   # 去除起点
            self.cost += eucl_dist(self.path[i], self.path[i+1])    # 计算两两节点之间的cost并累加


    def __repr__(self):
        return "<Path: %d elements, cost: %.3f: %s>" % (len(self.path), self.cost, self.path)

    def edges(self):
        return zip(self.path[0:-1], self.path[1:])

"""
Edge class
边类
"""

class Edge(object):
    def __init__(self, source, target, weight=1.0):
        self.source = source
        self.target = target
        self.weight = weight

    def __hash__(self):
        return hash("%s_%s_%f" % (self.source, self.target, self.weight))

    def __eq__(self, other):
        return self.source.state == other.source.state and self.target.state == other.target.state and self.weight == other.weight

    def __repr__(self):
        return "Edge(\n %r \n %r \n %r \n)" % (self.source, self.target, self.weight)

"""
Graph class
图类
"""

class Graph(object):

    def __init__(self, node_label_fn=None):
        self._nodes = list()    # 空列表存储节点
        self._edges = dict()    # 空字典存储图中的边  键为节点，值为与该节点相连的节点列表
        self.node_label_fn = node_label_fn if node_label_fn else lambda x: x    # 本方法使用默认的标签函数即标签函数是节点对象
        self.node_positions = dict()    # 空字典存储节点位置

    def __contains__(self, node):   # 判断图中是否包含指定的节点
        return node in self._nodes

    def add_node(self, node):
        """
        Adds a node to the graph
        添加一个节点到图中
        """
        # the function gets called when add_edge is called, so just check that we do not add several nodes
        # 函数会在调用add_edge时被调用，因此只需检查我们是否添加了多个节点即可
        if not node in self._nodes:
            self._nodes.append(node)   # 节点不在节点列表中就添加进去

    def add_edge(self, node1, node2, weight=1.0, bidirectional=False):
        """
        Purpose:  Adds an edge between node1 and node2
                  在节点1和节点2之间添加边
                  Adds the nodes to the graph first if they don't exist
                  如果节点不存在，则首先将它们添加到图中
        """
        self.add_node(node1)
        self.add_node(node2)

        node1_edges = self._edges.get(node1, list())
        node1_edges.append(Edge(node1, node2, weight))

        self._edges[node1] = node1_edges
        if bidirectional:
            node2_edges = self._edges.get(node2, set())
            node2_edges.add(Edge(node2, node1, weight))
            self._edges[node2] = node2_edges

    def remove_edge(self, node1, node2, bidirectional=False):
      """
      Purpose:  Removes an edge that connects id(node1) to id(node2) Will not look for if the costs of the nodes matches those in the edgelist from before, only checks for id on source and target
                移除连接节点1和节点2的边 不查看节点的代价是否与之前的边列表中的代价一致，只检查源节点和目标节点的 id
      :params:  node1, node2: two SearchNode objects
                bidirectional: tells if that edge was two-ways or not
                双向：说明该边是否为双向边
      """

      if node1 in self._edges:
          edgelist = self._edges[node1]
          for i in range(len(edgelist)):
              edge = edgelist[i]
              if edge.target == node2:
                  try:
                      self._edges[node1].remove(self._edges[node1][i])
                      # print("Successfully removed edge")
                  except:
                      print("Didn't find edge", id(node1), id(node2))
                  break

      if bidirectional:
        if node2 in self._edges:
          edgelist = self._edges[node2]
          for i in range(len(edgelist)):
              edge = edgelist[i]
              if edge.target == node1:
                  try:
                      self._edges[node2].remove(self._edges[node2][i])
                      # print("Successfully removed edge")
                  except:
                      print("Didn't find edge", id(node2), id(node1))
                  break

    def set_node_positions(self, positions):
        self.node_positions = positions

    def set_node_pos(self, node, pos):
        """Sets the (x,y) pos of the node, if it exists in the graph."""
        if not node in self:
            raise Exception('Node is noth in graph!')
        self.node_positions[node] = pos

    def get_node_pos(self, node):
        if not node in self:
            raise Exception('Node is noth in graph!')
        return self.node_positions[node]

    def node_edges(self, node):
        if not node in self:
            raise Exception('Node is noth in graph!')
        return self._edges.get(node, set())

    def updateEdges(self, node):
      """
      Purpose:  when a node has been rewired, its parent has changed. Therefore the list of edges originally in the graph extending from this node doesn't contain the correct source (since both its cost and parent has been updated). This function updates the source of all of those edges
                当节点被重新布线时，它的父节点也发生了变化。因此，图中最初从该节点延伸出来的边的列表不包含正确的来源（因为其代价和父节点都已更新）。此函数将更新所有这些边的源代码

      :params:  node: a SearchNode object
      """
      if node in self._edges:
        for edge in self._edges[node]:
          edge.source = node
      else:
        # the node might not have any edges connected to itself as it could be on the edge of a path
        # 节点可能没有任何边与自己相连，因为它可能位于路径的边上
        pass

"""
Ellipse class
椭圆类
informed rrt*
"""

class Ellipse(object):
  """
  A class used to change the sampling distribution over which the RRT star samples after it has found an initial goal. a and b will be the distances from the center of the ellipse out to the maximum in the horizontal and vertical direction, respectively. See https://en.wikipedia.org/wiki/Ellipse#In_Cartesian_coordinates and https://stackoverflow.com/a/46840451/10308389.
  a和b分别是水平和垂直方向上从椭圆中心到最大值的距离
  """
  def __init__(self, a=1, b=1, pos=(0, 0), orientation=0.0):
    """
    If not specified, Ellipse will be a unit circle centered in the origin
    如果未指定，椭圆将是一个以原点为中心的单位圆
    """
    self.a = a
    self.b = b

    if a < 0:
      self.a = -a
      print("Ellipse received negative a, setting it to -a")
    if b < 0:
      self.b = -b
      print("Ellipse received negative b, setting it to -b")

    self.pos = pos
    self.orientation = orientation

  def __eq__(self, other):
    return self != None and self.a == other.a and self.b == other.b and self.pos == other.pos and self.orientation == other.orientation

  def __repr__(self):
    return "Ellipse(%r,%r,%r,%r)" % (self.pos, self.a, self.b,self.orientation)

  def getCenterPos(self):
    return self.pos

  def getOrientation(self):
    return self.orientation

  def generateTheta(self):
    """
    Purpose:  Generate random theta within [-pi/2, 3pi/2]
    """
    u = random.random() / 4.0
    theta = np.arctan(self.b / self.a * np.tan(2*np.pi*u))
    v = random.random() # [0.,1.)
    if v < 0.25:
        return theta
    elif v < 0.5:
        return np.pi - theta
    elif v < 0.75:
        return np.pi + theta
    else:
        return -theta

  def getRadius(self,theta):
    """
    Purpose:  Calculate the distance from ellipse center given an angle
    """
    return self.a * self.b / np.sqrt((self.b*np.cos(theta))**2 + (self.a*np.sin(theta))**2)

  def randomPointOrigin(self):
    """
    Purpose:  generates a random point within the ellipse as if centered in the origin
    """
    rand_th = self.generateTheta()
    max_radius = self.getRadius(rand_th)
    rand_rad = max_radius * np.sqrt(random.random())
    return (rand_rad*np.cos(rand_th),rand_rad*np.sin(rand_th))

  def transformPoint(self, x, y):
    """
    Purpose:  Transform point in local coordinate system into global x,y
    """
    # First, skew according to orientation
    r = math.sqrt(x**2 + y**2)
    xnew = r * np.cos(self.orientation)
    ynew = r * np.sin(self.orientation)
    # Then, skew according to offset from origin
    xnew += self.pos[0]
    ynew += self.pos[1]
    return (xnew, ynew)

  def randomPoint(self):
    """
    Purpose:  Generates a random point within the ellipse considering its orientation and center position
    根据椭圆的方向和中心位置，在椭圆内随机生成一个点
    """
    x, y = self.randomPointOrigin()
    return self.transformPoint(x, y)

  def generateEllipseParams(self, path):
    """
    Purpose:  Takes in a path (list of (x,y)-tuples, and browses through all its SearchNodes to generate a, b, pos, and orientation for it self
    目的：接收路径（(x,y)图元列表），并浏览其所有搜索节点，为其生成 a、b、pos 和方向。
    """
    start_pos = path[0]
    goal_pos = path[-1]
    dx = goal_pos[0] - start_pos[0]
    dy = goal_pos[1] - start_pos[1]
    r = math.sqrt(dx**2 + dy**2)
    alpha = np.arctan2(dy, dx)

    x_center, y_center = (start_pos[0]+r/2*math.cos(alpha),start_pos[1]+r/2*math.sin(alpha))

    amax = eucl_dist(start_pos, goal_pos) / 2
    bmax = 0
    # normal vec is chosen to have norm 1 for simplicity
    # 为简单起见，选择法向量的常模为1
    normal_vec = [np.cos(alpha + math.pi/2), np.sin(alpha + math.pi/2)]
    ortho_vec = [np.cos(alpha + math.pi), np.sin(alpha + math.pi)]

    # margin is calculated as np.dot(theta,x) / ||theta||
    # margin 的计算公式为 np.dot(theta,x) / ||theta||
    # normal vector gives the distance 法向量给出了距离
    for i in range(0, len(path)):
      margin_a = math.fabs(dotProd(ortho_vec, path[i]))
      margin_b = math.fabs(dotProd(normal_vec, path[i]))

      if margin_b > bmax:
        bmax = margin_b

      if margin_a > amax:
        amax = margin_a

    self.a = amax * 0.7
    self.b = bmax * 1.05
    self.pos = (x_center, y_center)
    self.orientation = alpha

  def plot(self, ax, color="#BC3E23", increment=1):
    """
    Purpose:  Plots an Ellipse object onto Axes object

    :params:  ax is the Axes object from matplotlib
              ellipse is the Ellipse object
    """
    xc, yc = self.pos
    th = self.orientation
    x = []
    y = []
    for deg in range(0,360,increment):
      radian = degToRad(deg)
      radius = self.getRadius(radian)
      x.append(xc + radius * math.cos(radian + th))
      y.append(yc + radius * math.sin(radian + th))

    ax.plot(x, y, color=color, linewidth=3)