import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.animation as pathces
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from numpy.typing import ArrayLike
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import seaborn as sns

class Animate:
    def __init__(self, sol, N, x_g, x_0, x_o, r_o, poly, accs, time_step, times, labels, normalized_velocity=True, max_nom_vel=0.4, **kwargs):
        self.sol = sol
        self.N = N
        self.x_g = x_g
        self.x_0 = x_0
        self.x_o = x_o
        self.r_o = r_o
        self.poly = poly
        self.fig = 0
        self.plot_arrows = True
        #self.root_childs = root_childs
        self.accs = accs
        self.arrows = np.array([])
        self.keys = []
        self.times = times
        self.time_step = time_step
        self.labels = labels
        self.logarthmic_arrows = False
        self.color_dict = {
            'md': 0,
            'ga': 1,
            'ca': 2,
            'oa': 3,
            'po': 3,
            'da': 4,
            'nn': 5,
            'lo': 6,
            'no': 7
        }
        self.cmap = matplotlib.cm.get_cmap('tab10')
        self.velocity_cmap = sns.color_palette("flare_r", as_cmap=True) # matplotlib.cm.get_cmap('plasma')
        self.normalized_velocity = normalized_velocity
        self.max_nom_vel = max_nom_vel # upper bound for normalizing velocity colors
        if not self.normalized_velocity:
            self.velocity_cmap.set_over((1.0,0.0,1.0))#self.velocity_cmap(1.0))
        for key, value in kwargs.items():
            setattr(self,key,value)

        self.agents, self.fig, ax = self.create_env()
        #self.agents = self.agents[0]

        if self.plot_arrows:
            self.init_arrows()
        
        
    
    def _logarithmic_arrows(self, dx, dy):
        if not self.logarthmic_arrows:
            return dx, dy
        else:
            norm = np.linalg.norm(np.array([dx,dy]))
            scale = np.log(norm+1)/norm
            dx = scale*dx
            dy = scale*dy
        return dx, dy

    def animate(self, i):
        nsteps = self.sol.y.shape[-1]
        #self.agents.set_xdata(self.sol.y[0: 2 * self.N: 2, i % nsteps])
        #self.agents.set_ydata(self.sol.y[1: 2 * self.N + 1: 2, i % nsteps])
        xs = self.sol.y[0: 2 * self.N: 2, i % nsteps].flatten()
        ys = self.sol.y[1: 2 * self.N + 1: 2, i % nsteps].flatten()
        positions = np.array(np.append(xs, ys))
        positions = positions.reshape(2, -1).T
        #print(np.shape(positions))
        #print(positions)
        self.agents.set_offsets(positions)
        
        if self.plot_arrows:
            l = 0
            j = 0
            
            #print(self.agents, self.arrows)
            for a in self.accs[1:]:
                xs = self.sol.y[0: 2 * self.N: 2, i % nsteps][j]
                ys = self.sol.y[1: 2 * self.N + 1: 2, i % nsteps][j]
                
                for key in a.keys():
                    dx = 0
                    dy = 0
                    acc_times = [item[1] for item in a[key]]
                    
                    #if i*self.time_step in acc_times:
                    if self.times[i] in acc_times:
                        index = acc_times.index(self.times[i])
                        dx = a[key][index][0][0]
                        dy = a[key][index][0][1]
                        dx, dy = self._logarithmic_arrows(dx,dy)
                        self.arrows[l].set_data(x=xs.item(), y=ys.item(), 
                                    dx=dx, 
                                    dy=dy)
                        self.arrows[l].set_alpha(1)
                    
                    else: 
                        self.arrows[l].set_alpha(0)

                    
                    #dx=a[key][i % nsteps][0][0]
                    #dy=a[key][i % nsteps][0][1]                  
                    l += 1
                j += 1
            

            r = np.concatenate([np.array([self.agents]), self.arrows])
            return r

        else:
            return self.agents, # Do not remove comma!

    def init(self):  # only required for blitting to give a clean slate.
        if self.plot_arrows:
            r = np.concatenate([np.array([self.agents]), self.arrows])
            return r

        else:
            return self.agents, # Do not remove comma!
    
    def init_arrows(self):
        arrow = 0
        i = 0
        
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#66CDAA"]
        labels = self.labels
        mem_color = [] # To memorize used color
        
        for a in self.accs[1:]:
            child_accs = a
            l = 0
            for key in child_accs.keys():
                x = self.sol.y[0: 2 * self.N: 2, 0][i].item()
                y = self.sol.y[1: 2 * self.N + 1: 2, 0][i].item()
                dx = 0#child_accs[key][0][0][0]
                dy = 0#child_accs[key][0][0][1]
                dx, dy = self._logarithmic_arrows(dx, dy)
                color = self.cmap(self.color_dict[key[0:2]])
                if colors[l%len(colors)] not in mem_color:
                    arrow = plt.arrow(x-dx, y-dy, dx, dy, 
                                        color = color,
                                        label = labels[key[0:2]],
                                        head_width=0.07)
                    mem_color.append(colors[l%len(colors)])
                else: 
                    arrow = plt.arrow(x-dx, y-dy, dx, dy, color = color,head_width=0.07)
                        
                l += 1
                self.arrows = np.append(self.arrows,arrow)
                self.keys.append(key)
            i += 1   
    
    def plot_polygon_obstacle(self, polygons, ax):
        label = 'Obstacle'
        i = 0
        for poly in polygons:
            patch1 = matplotlib.patches.Polygon(poly.exterior.coords, label=label if i ==0 else None,# + str(i)
                                                fill=False)
            ax.add_patch(patch1)
            i += 1

    def create_env(self, frame=0, fignum=None):
        """
        Initialize the environment figure with starting positions, obstacles, goals and traces of the traversed paths.
        The traces are colored according to velocity.
        """
        #plt.figure()
        #fig = plt.gcf()
        #ax = plt.gca()
        # Create figure
        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, num=fignum)
        labels = ["Traversed Path", "Goal", "Start"]

        # Define data variables
        n_steps = len(self.sol.y[0])
        x_vels = [self.sol.y[2 * i+2*self.N] for i in range(self.N)]
        y_vels = [self.sol.y[2 * i + 1+2*self.N] for i in range(self.N)]
        x_dots = np.array([[np.linalg.norm([x_vels[i][j], y_vels[i][j]]) for j in range(n_steps)] for i in range(self.N)])
        v_max = np.amax(x_dots)
        x_dots_normed = x_dots * 1/v_max

        # Create map for normalizing colors of the traces
        self.scalar_map = matplotlib.cm.ScalarMappable(cmap = self.velocity_cmap)
        self.scalar_map.set_clim(vmin=0, vmax = v_max if self.normalized_velocity else self.max_nom_vel)

        # Plot traces of each drone
        for i in range(self.N):
            x_coords = self.sol.y[2 * i]
            y_coords = self.sol.y[2 * i + 1]

            # Plot an invisible trace for automatic resizing of the plotting area
            plt.plot(x_coords, y_coords, 'k,', zorder=1, alpha=0)

            # Define line segments of the colored trace
            points = np.array([x_coords, y_coords]).T.reshape(-1,1,2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            self.color_norm = plt.Normalize(0, v_max if self.normalized_velocity else self.max_nom_vel)
            lc = matplotlib.collections.LineCollection(segments, cmap=self.velocity_cmap, norm=self.color_norm)
            lc.set_array(x_dots[i])
            line = ax.add_collection(lc)

        # Plot goals and starting positions
        ax.plot(self.x_g[:, 0], self.x_g[:, 1], alpha=0)
        ax.plot(self.x_0[:, 0], self.x_0[:, 1], alpha=0)
        ax.scatter(self.x_g[:, 0], self.x_g[:, 1], color='g', marker=(5,0), label=labels[1], zorder=2)
        ax.scatter(self.x_0[:, 0], self.x_0[:, 1], color='r', marker=(4,0,45), label=labels[2], zorder=2)

        # Plot circular obstacles
        label = "Obstacle"
        for j in range(len(self.x_o)):
            if j>0:
                label = None
            circle = plt.Circle((self.x_o[j][0], self.x_o[j][1]), self.r_o[j], color='k', fill=False, label=label if self.poly is None else None)
            ax.add_artist(circle)

        # Plot initial agent positions
        agents = ax.scatter(self.sol.y[0: 2 * self.N: 2, frame], self.sol.y[1: 2 * self.N + 1: 2, frame], color=(0,0,0), marker=(4,2,45), label="Drone", zorder=2)
        
        # Plot obstacle polygons
        self.plot_polygon_obstacle(self.poly, ax)

        #ax.set_facecolor((0.85, 0.90, 0.95))

        # Resize the plotting area to include traces
        ax.relim()

        return agents, fig, ax

    def plot_frame(self, frame=0, md_ranges=[], lidars=False):
        agents, fig, ax = self.create_env(frame, fignum=f'simulation frame {frame}')
        # Add colorbar on top of the plotting area
        extend_cmap = 'neither' if self.normalized_velocity else 'max'
        divider = make_axes_locatable(fig.axes[0])
        cax = divider.append_axes("top", size="5%", pad=0.1) 
        fig.colorbar(matplotlib.cm.ScalarMappable(self.color_norm, self.velocity_cmap), shrink=0.75, extend=extend_cmap, cax=cax, label="Speed [m/s]", orientation='horizontal')
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')

        # Add max distance
        for i in range(len(md_ranges)):
            pos = agents.get_offsets()[i]
            for j in range(len(md_ranges[i])):
                md_circle = plt.Circle(pos, md_ranges[i][j], facecolor=self.cmap(j), 
                                    edgecolor=np.array(self.cmap(j))*0.25, alpha=0.075,
                                    label="Max distance")
                ax.add_artist(md_circle)

        # Add legend to the right of the plotting area
        lax = divider.append_axes("right", size="15%", pad=0.1)
        h,l = fig.axes[0].get_legend_handles_labels()
        fig.axes[0].legend(bbox_to_anchor=(0,0,1,1), bbox_transform=lax.transAxes)
        lax.axis("off")

        # Set axis labels
        fig.axes[0].set_xlabel('x [m]')
        fig.axes[0].set_ylabel('y [m]')

        # Resize figure
        fig.axes[0].relim()
        cax.relim() # not needed
        fig.axes[0].set_aspect('equal', 'box')
        fig.axes[0].autoscale_view()
        plt.tight_layout()
        fig.subplots_adjust(right=0.75)
        return fig, ax



class OldAnimate:
    def __init__(self, sol, N, x_g, x_0, x_o, r_o, poly, accs, time_step, times, labels, **kwargs):
        self.agents = 0
        self.i = 0
        self.sol = sol
        self.N = N
        self.x_g = x_g
        self.x_0 = x_0
        self.x_o = x_o
        self.r_o = r_o
        self.fig = 0
        self.agents, self.fig, ax = self.create_env()
        self.agents = self.agents[0]



    def animate(self,i):
        nsteps = self.sol.y.shape[-1]
        #x1,x2 = sol.y[0: 2 * N: 2, i % nsteps]
        #y1,y2 = sol.y[1: 2 * N + 1: 2, i % nsteps]
        #print(np.linalg.norm(np.array([x1,y1])-np.array([x2,y2])))
        self.agents.set_xdata(self.sol.y[0: 2 * self.N: 2, i % nsteps])
        self.agents.set_ydata(self.sol.y[1: 2 * self.N + 1: 2, i % nsteps])
        return self.agents,

    def init(self):  # only required for blitting to give a clean slate.
        return self.agents,
    
    def create_env(self):
        plt.figure()
        fig = plt.gcf()
        ax = plt.gca()
        for i in range(self.N):
            plt.plot(self.sol.y[2 * i], self.sol.y[2 * i + 1], 'y--')
        plt.plot(self.x_g[:, 0], self.x_g[:, 1], 'go')
        plt.plot(self.x_0[:, 0], self.x_0[:, 1], 'ro')

        plt.axis(np.array([-12, 12, -12, 12]))
        plt.gca().set_aspect('equal', 'box')

        #fig = plt.gcf()
        #ax = plt.gca()
        
        for j in range(len(self.x_o)):
            print(self.x_o[j][0], self.x_o[j][1])
            circle = plt.Circle((self.x_o[j][0], self.x_o[j][1]), self.r_o[j], color='k', fill=False)
            plt.gca().add_artist(circle)

        agents = plt.plot(self.sol.y[0: 2 * self.N: 2, 0], self.sol.y[1: 2 * self.N + 1: 2, 0], 'ko')
        
        return agents, fig, ax







# --------------------------------------------

def show_animation(A, sol, save_path, save_animation=False, metadata={}, name="rmp_our_example"):
    # Initialize animation object
    ani_saveable = animation.FuncAnimation(
        A.fig, A.animate, init_func=A.init, frames=sol.y.shape[-1], 
        interval=30, blit=True, repeat=True)
    
    # Add colorbar on top of the plotting area
    extend_cmap = 'neither' if A.normalized_velocity else 'max'
    divider = make_axes_locatable(A.fig.axes[0])
    cax = divider.append_axes("top", size="5%", pad=0.1) 
    A.fig.colorbar(matplotlib.cm.ScalarMappable(A.color_norm, A.velocity_cmap), shrink=0.75, extend=extend_cmap, cax=cax, label="Speed [m/s]", orientation='horizontal')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')

    # Add legend to the right of the plotting area
    lax = divider.append_axes("right", size="15%", pad=0.1)
    h,l = A.fig.axes[0].get_legend_handles_labels()
    A.fig.axes[0].legend(bbox_to_anchor=(0,0,1,1), bbox_transform=lax.transAxes)
    lax.axis("off")

    # Set axis labels
    A.fig.axes[0].set_xlabel('x [m]')
    A.fig.axes[0].set_ylabel('y [m]')

    # Resize figure
    A.fig.axes[0].relim()
    cax.relim() # not needed
    A.fig.axes[0].set_aspect('equal', 'box')
    A.fig.axes[0].autoscale_view()
    plt.tight_layout()
    A.fig.subplots_adjust(right=0.75)

    # Show figure
    plt.show()
    # Save
    if save_animation:
        print("Saving animation")
        ani_saveable.save(save_path / f'{name}.gif', writer='pillow', fps=30,dpi=60, metadata=metadata)
        print("Saving done")

# --------------------------------------------

def plot_acceleration(acc_dict, save_fig=False, save_path="", labels = dict(), metadata=None, logy=False, interval=0.2):
    fig, ax = plt.subplots()

    #ts = np.arange()
    legends = np.array([])
    for i in acc_dict.keys():
        legends = np.append(legends,labels[i[0:2]])
        norm_temp = []
        for j in acc_dict[i]:
            norm_temp.append(np.linalg.norm(j[0]))
        ts = np.arange(len(norm_temp))*interval
        p = ax.plot(ts, norm_temp)
        #color = p[0].get_color()
        #print(color)
    if logy:
        ax.set_yscale('log')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Acceleration [m/s$^2$]")

    divider = make_axes_locatable(ax)
    lax = divider.append_axes("right", size="15%", pad=0.1)
    lgd = ax.legend(legends, bbox_to_anchor=(0,0,1,1), bbox_transform=lax.transAxes)
    
    ax.relim()
    ax.set_aspect('auto','box')
    ax.autoscale_view()
    #fig.subplots_adjust(right=0.85)
    fig.tight_layout()
    lax.axis("off")

    if(save_fig):
        plt.savefig(save_path, metadata=metadata, bbox_inches='tight', bbox_extra_artists=[lgd])

# Construct rmp graph for plot 
def construct_rmp_graph(node, graph=None, layer_i=None):
    """
    Recursively construct a graph of the RMP tree

    node: RMP node that forms the root of the graph
    graph: networkx graph object
    layer_i: the layer that the current node is at
    """
    if layer_i == None:
        layer_i = 0
    else:
        layer_i +=1
    if graph==None:
        graph = nx.Graph()
        #graph = nx.complete_multipartite_graph()
        graph.add_node(node.name, layer=layer_i)
        layer_i += 1
    
    for child in node.children:
        graph.add_node(child.name, layer=layer_i)
        graph.add_edge(node.name, child.name)
        construct_rmp_graph(child, graph, layer_i)
    return graph

# Plot rmp graph
def plot_rmp_tree(node, graph=None, save_fig=False, save_path="./", ):
    rmp_graph = construct_rmp_graph(node, graph)
    plt.figure()
    #pos = graphviz_layout(rmp_graph)
    pos = nx.multipartite_layout(rmp_graph, subset_key="layer")
    nx.draw_networkx(rmp_graph,pos)
    if save_fig:
        plt.savefig(save_path / 'rmp_tree_graph')

# plot difference plot for ttest

def plot_diff_ttest(baseline: ArrayLike, diffs: ArrayLike, label: str):
    fig, ax = plt.subplots()
    ax.plot(baseline, diffs,'o', label=label)
    ax.set_title(f"Difference plot of {label}")
    ax.set_xlabel("Baseline")
    ax.set_ylabel("Difference")
    return fig, ax