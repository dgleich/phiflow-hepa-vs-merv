"""
Solves the incompressible Navier-Stokes equations in conjunction with
an advection equation in a closed box with a filtration system. 
Strongly influenced by the Phiflow smoke test demo. 


    +----------------------------------------------+
    |                    XX                        |
    |                 <-|  |Mixing                 |
    |                 <-|  |fan                    |
    |                 <-|  |                       |
    |                 <-|  |                       |
    |                    XX                        |
    |                                              |
    |                                              |
    |                                              |
    |                           ^ ^ ^ ^ ^          |
    |                           | | | | |          |
    |                         -------------        |
    |           _            X  fan+filter X       |
    |          / \            -------------        |
    |         | S |                                |
    |          \_/                                 |
    |                                              |
    +----------------------------------------------+
"""

# set up the code to make it easy to switch. 
if True: 
    NPTS = 200
    NSMOKEPTS = 400 
    dt = 0.2
else: 
    NPTS = 100
    NSMOKEPTS = 200 
    dt = 1
TSMOKE = 10 
TFILTER = 40
TMAX = 250 
MIXFANVELOCITY = 10
DIFFUSION = 2
CRFANVELOCITY = 9 
CRFANFILTER = 0.7
HEPAFANVELOCITY = 3.5 #6.3 #3
HEPAFANFILTER = 0.999
STARTVAL = 1000/(NSMOKEPTS*NSMOKEPTS*2)

#from phi.jax import flow
from phi import flow 
import matplotlib.pyplot as plt
from tqdm import tqdm
import phi
import math
from matplotlib.patches import Rectangle
import csv

import numpy as np

def create_fields():
    velocity = flow.StaggeredGrid(
        values=(0.0, 0.0),
        extrapolation=0.0,
        x=2*NPTS,
        y=NPTS,
        bounds=flow.Box(x=2*100, y=100),
    )
    smoke = flow.CenteredGrid(
        values=STARTVAL,
        extrapolation=flow.extrapolation.BOUNDARY,
        x=2*NSMOKEPTS,
        y=NSMOKEPTS,
        bounds=flow.Box(x=2*100, y=100),
    )
    return velocity, smoke

def create_insmoke(smoke):
    insmoke = phi.field.resample(
                    flow.Sphere(
                        x=175,
                        y=50,
                        radius=5,
                    ), 
                    to=smoke, 
                    soft=True)
    # scale to insure constant inflow for different grid sizes                 
    insmoke *= (200*400)/(NSMOKEPTS*NSMOKEPTS*2)
    return insmoke 

def set_zero_value(field, geometry):
    return field - field*phi.field.resample(geometry, to=field, soft=True)


def create_fan_geometry(xstart, xend, ystart, yend, velocity):
    # check for the orientation... 
    xsize = xend-xstart
    ysize = yend-ystart
    fanbox = flow.Box(
        x=(xstart,xend), 
        y=(ystart,yend)
    )
    fanboundary = phi.physics.fluid.Obstacle(fanbox, velocity=velocity)
    return fanboundary, fanbox 

def simulation(fanvelocity, filtration): 
    velocity, smoke = create_fields()
    inflow = create_insmoke(smoke)*dt 
    diffusion = DIFFUSION*smoke.with_values(1.0)

    boundaries = [] 
    mixing_fan_boundary, mixing_fan_box = create_fan_geometry(92, 108, 60, 85, (MIXFANVELOCITY, 0))
    boundaries.append(mixing_fan_boundary)
    mixing_fan_boundary, mixing_fan_box = create_fan_geometry(92, 108, 15, 40, (-MIXFANVELOCITY, 0))
    boundaries.append(mixing_fan_boundary)
    filter_fan_boundary, filter_fan_box = create_fan_geometry(25, 50, 42, 58, (0, fanvelocity))
    boundaries.append(filter_fan_boundary)

    
    

    # set the filtersize so we filter the volume
    # of air that moves through the fan in one timestep. 
    #filtersize = CRFANVELOCITY*dt 
    filtersize = fanvelocity*dt 
    if filtersize >= 16:
        # throw an error if the filtersize is too large
        raise ValueError("Filtersize too large, reduce filter fan velocity or timestep size.")
    filterinterface = flow.Box(
       #x = (154,171),
       x = (29, 46),
       y = (50,50+filtersize)
    )
    filterfield = phi.field.resample(filterinterface, to=smoke, soft=True)

    solver = phi.math.Solve('CG', max_iterations=5000, abs_tol=1e-5)

    # set no diffusion within the filter fan box to avoid impacts of diffusion on the filter
    diffusion = set_zero_value(diffusion, filter_fan_box)

    # compute the effective filter field in light of the particle diffusion
    dfilterfield = flow.diffuse.implicit(filterfield, diffusion, dt, solve=solver)
    dfilterfield = dfilterfield + filterfield 
    dfilterfield = dfilterfield.with_values(phi.math.clip(dfilterfield.values, 0, 1))

    
    def step_pollute(velocity_prev, smoke_prev, dt):
        smoke_next = flow.advect.mac_cormack(smoke_prev, velocity_prev, dt) + inflow
        buoyancy_force = smoke_next * (0.0, 0.1) @ velocity
        smoke_next = flow.diffuse.implicit(smoke_next, diffusion, dt, solve=solver)
        velocity_tent = flow.advect.semi_lagrangian(velocity_prev, velocity_prev, dt) + buoyancy_force * dt
        velocity_tent = phi.physics.fluid.apply_boundary_conditions(velocity_tent, boundaries)
        velocity_next, pressure = flow.fluid.make_incompressible(velocity_tent, boundaries, solve=solver)
        return velocity_next, smoke_next
    
    def step_mix(velocity_prev, smoke_prev, dt):
        smoke_next = flow.advect.mac_cormack(smoke_prev, velocity_prev, dt, correction_strength=0.8)
        smoke_next = flow.diffuse.implicit(smoke_next, diffusion, dt, solve=solver)

        velocity_tent = flow.advect.semi_lagrangian(velocity_prev, velocity_prev, dt)
        velocity_tent = phi.physics.fluid.apply_boundary_conditions(velocity_tent, boundaries)
        velocity_next, pressure = flow.fluid.make_incompressible(velocity_tent, boundaries,  solve=solver)
        return velocity_next, smoke_next
    
    def step_filter(velocity_prev, smoke_prev, dt):
        smoke_next = flow.advect.mac_cormack(smoke_prev, velocity_prev, dt, correction_strength=0.8)

        smoke_removal = filtration * smoke_next * dfilterfield
        smoke_next = (smoke_next - smoke_removal)
        smoke_clipped = phi.math.clip(smoke_next.values, 0, math.inf)
        smoke_next = smoke_next.with_values(smoke_clipped)

        smoke_next = flow.diffuse.implicit(smoke_next, diffusion, dt, solve=solver)

        velocity_tent = flow.advect.semi_lagrangian(velocity_prev, velocity_prev, dt)
        velocity_tent = phi.physics.fluid.apply_boundary_conditions(velocity_tent, boundaries)
        velocity_next, pressure = flow.fluid.make_incompressible(velocity_tent, boundaries,  solve=solver)
        return velocity_next, smoke_next
    
    
    t = 0.0 
    while t <= TMAX: 
        if t <= TSMOKE: 
            velocity, smoke = step_pollute(velocity, smoke, dt)
        elif t <= TFILTER: 
            velocity, smoke = step_mix(velocity, smoke, dt)
        else: 
            velocity, smoke = step_filter(velocity, smoke, dt)
        
        yield (t, smoke, velocity)
        t += dt 

def update_history(t, smoke, smoke_hist):
    new_value = np.sum(smoke.values.numpy("y,x"))
    smoke_hist.append((t,new_value))
    return smoke_hist


def add_box(ax, bottom_left, width, height, edge_color='r'):
    rect = Rectangle(bottom_left, width, height, linewidth=2, edgecolor=edge_color, facecolor='none')
    ax.add_patch(rect)


def add_string(ax, str, bottom_left, width, height, color): 
    # Calculate the center of the rectangle
    center_x = bottom_left[0] + width / 2
    center_y = bottom_left[1] + height / 2
    ax.text(center_x, center_y, str, ha='center', va='center', color=color)

def plot_boxes(psmoke):
    mixing_fan_boxes = [((92, 60), 16, 25), ((92, 15), 16, 25)]
    filter_fan_box = ((25, 42), 25, 16)  # Assuming a single filter fan box for simplicity
    filter_interface_box = ((25+4, 50), 17, 1)  # Adjust 'filtersize' as per your calculation

    # ←↑→⟶
    #for box in mixing_fan_boxes:
    add_box(psmoke, *(mixing_fan_boxes[0]), edge_color='black')
    add_string(psmoke, "→", *(mixing_fan_boxes[0]), 'black')

    add_box(psmoke, *(mixing_fan_boxes[1]), edge_color='black')
    add_string(psmoke, "←", *(mixing_fan_boxes[1]), 'black')

    # Add rectangle for filter fan
    add_box(psmoke, *filter_fan_box, edge_color='white')
    add_string(psmoke, "↑             ", *filter_fan_box, "white")

    # Add rectangle for filter interface
    add_box(psmoke, *filter_interface_box, edge_color='white')
    
def update_plot(t, smoke, velocity, psmoke, phistory, smoke_hist, alt_hist):    
    smoke_values_extracted = smoke.values.numpy("y,x")
    extent = [0, 200, 0, 100]
    #vrange = math.floor(math.log10(STARTVAL))
    vrange = math.log10(STARTVAL)-0.5
    psmoke.imshow(np.log10(smoke_values_extracted), origin="lower", vmin=vrange, vmax=vrange+1.5, extent=extent)
    plot_boxes(psmoke)
    phistory.plot(*zip(*smoke_hist))
    phistory.plot(*zip(*alt_hist), linewidth=1, color="grey", zorder=-1)

    last_point = smoke_hist[-1]  # Get the last point
    x, y = last_point  # Unpack the x and y coordinates
    
    # Plot the last point as a distinct marker
    phistory.plot(x, y, 'wo')  # 'wo' for white circle
    phistory.text(x, y-100, f'{y:.2f} ', color='white', verticalalignment='top', horizontalalignment='right')

    # add the time value
    phistory.text(x, 0, f'time {t:.2f} ', color='darkgrey', verticalalignment='bottom', horizontalalignment='right')

def setup_plot(psmoke, phistory, type):
    psmoke.set_title(f"{type} smoke (log-scaled)")
    psmoke.axis("off")
    phistory.set_title(f"{type} total smoke history")
    phistory.set_ylim(000, 5500)
    phistory.axis("off")

    #phistory.set_yscale("log")

def main():

    plt.style.use("dark_background")
    

    sim_cr = simulation(CRFANVELOCITY, CRFANFILTER)
    sim_hepa = simulation(HEPAFANVELOCITY, HEPAFANFILTER)

    crhist = []
    hepahist = [] 

    pbar = tqdm(total=TMAX)

    f = plt.figure(figsize=[8,5])

    csv_file_path = 'output.csv'
    frame=0
    with open(csv_file_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        while True:
            try:
                t_cr, smoke_cr, velocity_cr = next(sim_cr)
                t_hepa, smoke_hepa, velocity_hepa = next(sim_hepa)
            except StopIteration:
                break 

            ((p_smoke_cr, p_history_cr), (p_smoke_hepa, p_history_hepa)) = f.subplots(2, 2, gridspec_kw={'width_ratios': [3, 1]})
            f.subplots_adjust(wspace=0)

            setup_plot(p_smoke_cr, p_history_cr, "MERV13")
            setup_plot(p_smoke_hepa, p_history_hepa, "HEPA")
            update_history(t_cr, smoke_cr, crhist)
            update_history(t_hepa, smoke_hepa, hepahist)
            update_plot(t_cr, smoke_cr, velocity_cr, p_smoke_cr, p_history_cr, crhist, hepahist)
            update_plot(t_hepa, smoke_hepa, velocity_hepa, p_smoke_hepa, p_history_hepa, hepahist, crhist)    

            p_smoke_cr.text(0, 50, f"Filtration {CRFANFILTER*100:.1f}% \nVelocity {CRFANVELOCITY} ",   color='darkgrey', verticalalignment='center', horizontalalignment='right')
            p_smoke_hepa.text(0, 50, f"Filtration {HEPAFANFILTER*100:.1f}% \nVelocity {HEPAFANVELOCITY} ",   color='darkgrey', verticalalignment='center', horizontalalignment='right')

            if t_cr <= TSMOKE:
                status = "Injecting smoke"
            elif t_cr <= TFILTER:
                status = "Mixing"
            else:
                status = "Filtering"
            p_smoke_hepa.text(0, 00, f"\n{status}", color='darkgrey', verticalalignment='top', horizontalalignment='left')

            last_cr = crhist[-1]  # Get the last entry from crhist
            last_hepa = hepahist[-1]  # Get the last entry from hepahist
            csvwriter.writerow([t_cr, last_cr[1], last_hepa[1]])

            #f.tight_layout()    

            plt.draw()
            f.savefig("frame-"+str(frame)+".png", dpi=300)
            frame += 1
            plt.pause(0.01)
            plt.clf()

            pbar.update(dt)





""" Helper function if we want to visualize the magnitude of the velocity field."""    
def get_velocity(velocity_field):
    #pressure_extracted = pressure.values.numpy("y,x")
    # Extract the x and y components
    vel_x = velocity_field.values['x'].numpy('y,x')
    vel_y = velocity_field.values['y'].numpy('y,x')
    N = vel_x.shape[0]  # or another appropriate value depending on your grid shape
    vel_x_centered = 0.5 * (vel_x[:-1, :] + vel_x[1:, :])
    vel_y_centered = 0.5 * (vel_y[:, :-1] + vel_y[:, 1:])
    # Compute the magnitude (L2 norm) for each cell. 
    # The last axis typically corresponds to the vector components.
    # Compute magnitude
    magnitude = np.sqrt(vel_x_centered**2 + vel_y_centered**2)     
    return magnitude   

if __name__ == "__main__":
    main()
    #simulation(CRFANVELOCITY, CRFANFILTER)
    #simulation(HEPAFANVELOCITY, HEPAFANFILTER)
