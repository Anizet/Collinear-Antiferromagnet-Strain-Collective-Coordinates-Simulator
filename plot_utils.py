from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.animation as animation
from tqdm import tqdm
import re
import mplcyberpunk as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def single_plot(self,color="black", x = "x", label = None):
        if x == "x": 
            plt.plot(np.array(self.time[::1])*1e9, np.array(self.positions[::1])*1e6, linewidth=2, color = color, label = label) # label = rf'$\varepsilon_0$ = {self.strain:.2e}'
            plt.ylabel(r'Position ($\mu m$)', fontsize = 18)

        if x == "v": 
            plt.plot(np.array(self.time[::1])*1e9, np.array(self.velocities[::1]), linewidth=2 , color = color, label = label)
            plt.ylabel(r'Velocity ($\frac{m}{s}$)', fontsize = 18)

        if x == "a": 
            plt.plot(np.array(self.time[::1])*1e9, np.array(self.accelerations[::1]), linewidth=2, color = color, label = label)
            plt.ylabel(r'Acceleration ($\frac{m}{s^2}$)', fontsize = 18)
        plt.xlabel('Time (ns)', fontsize = 18)
        plt.xlim(0,self.Time*1e9)
        plt.yticks(fontsize = 16)
        plt.xticks(fontsize = 16)
        plt.legend(loc = 'upper left')

#def x_v_a_plot(self, data = False, offset = 0.0, vterm = False, sk = 5, oscF = None):
#        if data == True:
#            Tk().withdraw()  # Close the root window
#            filename = askopenfilename(title="Select the .npy file")
#            print(f"Selected file: {filename}")
#
#            DW_df = np.load(filename)
#            texp = DW_df[0]
#            DW_pos = DW_df[1]
#            DWpf = np.polyfit(texp, DW_pos, 10)
#            DW_fit = np.polyval(DWpf, texp)
#            vel_coeff = np.polyder(DWpf)
#            velocity_estimate = np.polyval(vel_coeff, texp)*1e9
#            accel_coeff = np.polyder(vel_coeff)
#            acceleration_estimate = np.polyval(accel_coeff, texp)*1e18
#
#        plt.figure(figsize=(12, 15), constrained_layout=True)
#        plt.subplot(3, 2, 1)
#        ax = plt.gca()  # get current axes
#
#        # Plot the background gradient first (so lines appear on top)
#        if getattr(self, "notch", False):
#            y0 = (self.pin0 - self.wnotch)*1e6
#            y1 = (self.pin0 + self.wnotch)*1e6
#
#            xlim = ax.get_xlim()  # current x-limits
#
#            # Create vertical gradient: max at center, light at edges
#            npts = 200
#            weights = 0.2 + 0.8 * (1 - np.abs(np.linspace(-1, 1, npts)))
#            gradient = weights.reshape(-1, 1)
#
#            ax.imshow(
#                np.tile(gradient, (1, 2)),
#                extent=[xlim[0], xlim[1], y0, y1],
#                origin="lower",
#                cmap="Blues",
#                alpha=0.6,
#                aspect="auto",
#                zorder=0
#            )
#            ax.set_xlim(xlim)
#
#        # Plot the data curves on top of gradient
#        if data:
#            ax.plot(texp[::sk], DW_pos[::sk]*1e6, 'd',
#                    label="Mumax+", color="darkorange", zorder=6)
#        ax.plot(np.array(self.time)*1e9, np.array(self.positions)*1e6 + offset,
#                '-', label='Model', solid_capstyle='round',
#                color='teal', zorder=5)
#        if oscF is not None:
#             ax.plot(np.array(self.time)*1e9, np.array(oscF)*1e6 + offset, label="No Rel. Force", color="#c67212", zorder=5)
#        # Labels and grid
#        ax.set_xlabel('Time (ns)')
#        ax.set_ylabel(r'Position ($\mu m$)')
#        ax.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
#
#        # Add gradient legend entry if notch exists
#        handles, labels = ax.get_legend_handles_labels()
#        if getattr(self, "notch", False):
#            gradient_patch = Patch(facecolor="blue", alpha=0.6, label="Strongest pinning region")
#            handles.append(gradient_patch)
#
#        ax.legend(handles=handles, frameon=False)
#        plt.tight_layout()
#
#        ax2 = plt.subplot(3, 2, 2)
#        #if data == True:
#        #    ax2.plot(texp[::sk], velocity_estimate[::sk], 'd', markersize=10, label="Mumax+", color='darkorange')
#        ax2.plot(np.array(self.time[::5])*1e9, self.velocities[::5], '-', label="Model", color='teal',zorder = 0)
#        if vterm == True:
#            ax2.plot(np.array(self.time[::1])*1e9, self.vtermstore[::1], ':', linewidth=3.5, 
#                     label=r"$V_{term}$", solid_capstyle='round', color = "#D50079")
#
#        ax2.set_xlabel('Time (ns)')
#        ax2.set_ylabel(r'Velocity ($\frac{m}{s}$)')
#        ax2.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
#        ax2.legend(loc='upper left',            # position
#                    frameon=True,                # keep frame
#                    edgecolor='black',           # border color
#                    fontsize=17,                 # font size
#                    shadow=True,                 # adds subtle shadow
#                    )
#        plt.tight_layout()
#        
#        # --- Inset ---
#        # ax_inset = inset_axes(ax2, width="40%", height="40%", loc='center right', borderpad = 1, bbox_to_anchor=(-0.05, -0.05, 1, 1),bbox_transform=ax2.transAxes,)
#        # if data == True:
#            # ax_inset.plot(texp[::sk], velocity_estimate[::sk], 'd', markersize=6, color='darkorange')
#        # ax_inset.plot(np.array(self.time[::5])*1e9, self.velocities[::5], '-', linewidth=2, color='teal',zorder = 0)
#        # if vterm == True:
#            # ax_inset.plot(np.array(self.time[::1])*1e9, self.vtermstore[::1], 'm:', linewidth=2.5)
## 
#        # ax_inset.set_ylim(-395, -382)
#        # ax_inset.set_xlim(0,1)  # keep the same x range as main plot
#        # ax_inset.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
## 
#        # mark_inset(ax2, ax_inset, loc1=3, loc2=4, fc="none", ec="0.5", lw=1.2)
#
#        plt.tight_layout()
#        #if data == True: plt.plot(texp[::sk], velocity_estimate[::sk], 'd', markersize = 10, label = "Mumax+", color = 'darkorange')  # Convert to nm
#        #plt.plot(np.array(self.time[::5])*1e9, self.velocities[::5], '-', linewidth=3, label = "Model", color = 'teal')
#        #if vterm == True : plt.plot(np.array(self.time[::1])*1e9, self.vtermstore[::1], 'm:', linewidth=3, label = r"$V_{term}$", solid_capstyle='round')
#        ##plt.title('Velocity vs Time')
#        #plt.xticks()
#        #plt.yticks()
#        #plt.xlabel('Time (ns)')
#        #plt.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
#        #plt.legend(frameon=False)
#        #plt.ylabel(r'Velocity ($\frac{m}{s}$)')
#        #plt.tight_layout()
#
#        #plt.figure(figsize=(11, 17))
#        #plot the acceleration
#        ax3 = plt.subplot(3, 2, 3)
#        #if data == True : ax3.plot(texp[::sk], acceleration_estimate[::sk], 'd', markersize = 10, label = "Mumax+", color = "darkorange")  # Convert to nm
#        ax3.plot(np.array(self.time[::5])*1e9, self.accelerations[::5], '-', color = "teal", label = "Model",zorder = 0)
#        #plt.plot(texp[::2], acceleration_estimate[::2], 'ro', markersize = 4)  # Convert to nm
#        #plt.title('Acceleration vs Time')
#        ax3.set_xlabel('Time (ns)')
#        ax3.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
#        #ax3.legend(frameon=False, loc = "lower left", bbox_to_anchor=(0.1, 0.0))
#        ax3.set_ylabel(r'Acceleration ($\frac{m}{s^2}$)')
#
#        # ax_inset2 = inset_axes(ax3, width="40%", height="40%", loc='center right', borderpad = 1.3,   bbox_to_anchor=(-0.025, 0.1, 1, 1),bbox_transform=ax3.transAxes,)
#        # if data == True:
#            # ax_inset2.plot(texp[::sk], acceleration_estimate[::sk], 'd', markersize=6, color='darkorange')
#        # ax_inset2.plot(np.array(self.time[::5])*1e9, self.accelerations[::5], '-', linewidth=2, color='teal',zorder = 0)
## 
#        # ax_inset2.set_ylim(-0.5e11,0.02e11)
#        # ax_inset2.set_xlim(0,1)  # keep the same x range as main plot
#        # ax_inset2.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
## 
#        # mark_inset(ax3, ax_inset2, loc1=1, loc2=2, fc="none", ec="0.5", lw=1.2)
#
#        plt.tight_layout()
#
#        plt.subplots_adjust(wspace=0.25)
#        
#        axf = plt.subplot(3,2,4)
#        axf.plot(np.array(self.time[::1])*1e9, np.array(self.Fformstore[::1])*1e15, color = 'teal',  label ='Magnetoelastic')
#        axf.plot(np.array(self.time[::1])*1e9, -np.array(self.Fvstore[::1])*1e15, color = "#D50079", label =r'$\dot{\chi}$ Damping')
#        axf.plot(np.array(self.time[::1])*1e9, np.array(self.Fdeltastore[::1])*1e15, color = "#2D65DF", label =r'$\dot{\Delta}$ Force')
#        #plt.title('Forces')
#        axf.legend(frameon=True,                # keep frame
#                    edgecolor='black',           # border color
#                    fontsize=16,                 # font size
#                    shadow=True,  
#                    loc = 'upper right'               # adds subtle shadow
#                    )
#        axf.set_xlabel('Time (ns)')
#        axf.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
#        axf.set_ylabel(r'Force ($fN$)')
#
#        # --- Inset ---
#        # ax_inset = inset_axes(
#            # axf,
#            # width="40%",        # width of inset
#            # height="40%",       # height of inset
#            # loc='upper right',  # anchor point
#            # bbox_to_anchor=(0, 0, 1, 1),  # relative bbox (full axes)
#            # bbox_transform=axf.transAxes,
#            # borderpad=1.4         # padding from the edge; increase to move inset down
#        # )        
#       ## Plot the same curves in the inset
#        # ax_inset.plot(np.array(self.time[::1])*1e9, np.array(self.Fformstore[::1])*1e15, color='teal')
#        # ax_inset.plot(np.array(self.time[::1])*1e9, -np.array(self.Fvstore[::1])*1e15, color="#D50079")
#        # ax_inset.plot(np.array(self.time[::1])*1e9, np.array(self.Fdeltastore[::1])*1e15, color="#2D65DF")
## 
#        ##Set the zoomed y-range
#        # ax_inset.set_ylim(-3.46, -3.43)
## 
#        ##Keep the same x-range as the main plot
#        # ax_inset.set_xlim(axf.get_xlim())
## 
#        ##Optional: grid for inset
#        # ax_inset.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
## 
#        ##Draw rectangle and connectors on main plot
#        # mark_inset(axf, ax_inset, loc1=3, loc2=4, fc="none", ec="0.5", lw=1.2)
#
#        plt.tight_layout()
#
#        #plt.figure(figsize=(11,5))
#        ax1 = plt.subplot(3,2,6)
#        # Plot domain wall width on primary y-axis
#        ax1.plot(np.array(self.time[::1])*1e9, np.array(self.Kstore[::1])*1e-3,
#                 color = "#2D65DF")
#        #ax1.set_title('Domain wall width')
#        ax1.set_xlabel('Time (ns)')
#        ax1.set_ylabel(r'K$_{eff}$ ($kJ/m^3$)', color="#2D65DF")
#        ax1.tick_params(axis='y', labelcolor="#2D65DF")
#        ax1.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
#        # Create second y-axis for strain
#        ax2 = ax1.twinx()
#        ax2.plot(np.array(self.time[::1])*1e9, np.array(self.strainstore[::1])*1e3,
#                 color = "teal")
#        ax2.set_ylabel(r'Strain $\cdot 1e-3$', color='teal')
#        ax2.tick_params(axis='y', labelcolor='teal')
#        # Combine legends
#        lines_1, labels_1 = ax1.get_legend_handles_labels()
#        lines_2, labels_2 = ax2.get_legend_handles_labels()
#        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', frameon = False)
#        ymin = np.min(np.array(self.Kstore[::1])*1e-3) - 0.5
#        ymax = np.max(np.array(self.Kstore[::1])*1e-3) + 0.5
#        ax1.set_ylim([ymin, ymax])
#        strain_min = np.min(np.array(self.strainstore[::1])*1e3)
#        strain_max = np.max(np.array(self.strainstore[::1])*1e3)
#
#        # Expand the range by a factor
#        center = 0.5 * (strain_max + strain_min)
#        half_range = 0.5 * (strain_max - strain_min)
#        scale_factor = 3.0  # Increase space 3x
#        ax2.set_ylim(center - scale_factor * half_range,
#             center + scale_factor * half_range)
#        plt.tight_layout()
#        #plt.subplot(1,3,1)
#        #plt.plot(np.array(self.time[::1])*1e9, np.array(self.deltastore[::1])*1e9, 'b-', linewidth=3, label ='Model')
#        #plt.plot(np.array(self.time[::1])*1e9, np.array(self.strainstore[::1])*1e3, 'g-', linewidth=3, label ='Strain [1e-3]')
#        #plt.title('Domain wall width')
#        #plt.xlabel('Time (ns)', fontsize = 16)
#        #plt.xticks(fontsize = 14)
#        #plt.yticks(fontsize = 14)
#        #plt.grid()
#        #plt.legend()
#        #plt.ylabel(r'$\Delta$ ($nm$)', fontsize = 16)
#        
#
#        plt.subplot(3,2,5)
#        plt.plot(np.array(self.time[::1])*1e9, np.array(self.deltastore[::1])*1e9, color = 'teal')
#        #plt.title('Effective Anisotropy')
#        plt.xlabel('Time (ns)')
#        plt.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
#        plt.ylabel(r'$\Delta$ ($nm$)')
#        plt.tight_layout()
#        plt.subplots_adjust(right=0.9)
#        
#        plt.figure(figsize=(18,4))
#        plt.subplot(1,3,1)
#        plt.plot(np.array(self.time[::1])*1e9, np.array(self.angle[::1]), color = 'teal', label ='Azimuthal angle [rad]')
#        plt.title('Azimuthal angle')
#        plt.xlabel('Time (ns)')
#        plt.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
#        plt.ylabel(r'$\phi$ (rad)')
#        plt.tight_layout()
#
#        plt.subplot(1,3,2)
#        plt.plot(np.array(self.time[::1])*1e9, np.array(self.wstore[::1]), color = 'teal', label ='Azimuthal angular frequency [rad/s]')
#        plt.title('Azimuthal angular frequency')
#        plt.xlabel('Time (ns)')
#        plt.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
#        plt.ylabel(r'$\dot{\phi}$ (rad/s)')
#        plt.tight_layout()
#
#        plt.subplot(1,3,3)
#        plt.plot(np.array(self.time[::1])*1e9, np.array(self.paccstore[::1]), color = 'teal', label =r'Azimuthal angular acc (rad/$s^2$)')
#        plt.title('Azimuthal angular acceleration')
#        plt.xlabel('Time (ns)')
#        plt.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
#        plt.ylabel(r'$\ddot{\phi}$ (rad/$s^2$)')
#        plt.tight_layout()
#        
#        self.plot_fft(np.array(self.angle))
#        
#        plt.subplots_adjust(wspace=0.1)
#        plt.show()

def x_v_a_plot(self, data = False, offset = 0.0, vterm = False, sk = 5, oscF=False):
        if data == True:
            Tk().withdraw()  # Close the root window
            filename = askopenfilename(title="Select the .npy file")
            print(f"Selected file: {filename}")
# 
            DW_df = np.load(filename)
            texp = DW_df[0]
            DW_pos = DW_df[1]
            DWpf = np.polyfit(texp, DW_pos, 10)
            DW_fit = np.polyval(DWpf, texp)
            vel_coeff = np.polyder(DWpf)
            velocity_estimate = np.polyval(vel_coeff, texp)*1e9
            accel_coeff = np.polyder(vel_coeff)
            acceleration_estimate = np.polyval(accel_coeff, texp)*1e18
# 
        plt.figure(figsize=(12, 15), constrained_layout=True)
        plt.subplot(3, 2, 1)
        ax = plt.gca()  # get current axes
# 
        # Plot the background gradient first (so lines appear on top)
        if getattr(self, "notch", False):
            y0 = (self.pin0 - self.wnotch)*1e6
            y1 = (self.pin0 + self.wnotch)*1e6
# 
            xlim = ax.get_xlim()  # current x-limits
# 
            # Create vertical gradient: max at center, light at edges
            npts = 200
            weights = 0.2 + 0.8 * (1 - np.abs(np.linspace(-1, 1, npts)))
            gradient = weights.reshape(-1, 1)
# 
            ax.imshow(
                np.tile(gradient, (1, 2)),
                extent=[xlim[0], xlim[1], y0, y1],
                origin="lower",
                cmap="Blues",
                alpha=0.6,
                aspect="auto",
                zorder=0
            )
            ax.set_xlim(xlim)
# 
        # Plot the data curves on top of gradient
        if data:
            ax.plot(texp[::sk], DW_pos[::sk]*1e6, '-', markersize=10,
                    label="Mumax+", color="darkorange", zorder=6)
        ax.plot(np.array(self.time)*1e9, np.array(self.positions)*1e6 + offset,
                '-', label='Model', solid_capstyle='round',
                color='teal', zorder=5)
# 
        # Labels and grid
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel(r'Position ($\mu m$)')
        ax.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
# 
        # Add gradient legend entry if notch exists
        handles, labels = ax.get_legend_handles_labels()
        if getattr(self, "notch", False):
            gradient_patch = Patch(facecolor="blue", alpha=0.6, label="Strongest pinning region")
            handles.append(gradient_patch)
# 
        ax.legend(handles=handles, frameon=False)
        plt.tight_layout()
# 
        ax2 = plt.subplot(3, 2, 2)
        if data == True:
            ax2.plot(texp[::sk], velocity_estimate[::sk], 'd', markersize=10, label="Mumax+", color='darkorange')
        ax2.plot(np.array(self.time[::5])*1e9, self.velocities[::5], '-', label="Model", color='teal',zorder = 0)
        if vterm == True:
            ax2.plot(np.array(self.time[::1])*1e9, self.vtermstore[::1], 'm:', linewidth=3.5, 
                     label=r"$V_{term}$", solid_capstyle='round')
# 
        ax2.set_xlabel('Time (ns)')
        ax2.set_ylabel(r'Velocity ($\frac{m}{s}$)')
        ax2.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
        ax2.legend(frameon=False)
        plt.tight_layout()
        # 
        # --- Inset ---
        ax_inset = inset_axes(ax2, width="40%", height="40%", loc='center right', borderpad = 1, bbox_to_anchor=(-0.05, -0.05, 1, 1),bbox_transform=ax2.transAxes,)
        if data == True:
            ax_inset.plot(texp[::sk], velocity_estimate[::sk], 'd', markersize=6, color='darkorange')
        ax_inset.plot(np.array(self.time[::5])*1e9, self.velocities[::5], '-', linewidth=2, color='teal',zorder = 0)
        if vterm == True:
            ax_inset.plot(np.array(self.time[::1])*1e9, self.vtermstore[::1], 'm:', linewidth=2.5)
# 
        ax_inset.set_ylim(-395, -382)
        ax_inset.set_xlim(0,1)  # keep the same x range as main plot
        ax_inset.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
# 
        mark_inset(ax2, ax_inset, loc1=3, loc2=4, fc="none", ec="0.5", lw=1.2)
# 
        plt.tight_layout()
        #if data == True: plt.plot(texp[::sk], velocity_estimate[::sk], 'd', markersize = 10, label = "Mumax+", color = 'darkorange')  # Convert to nm
        #plt.plot(np.array(self.time[::5])*1e9, self.velocities[::5], '-', linewidth=3, label = "Model", color = 'teal')
        #if vterm == True : plt.plot(np.array(self.time[::1])*1e9, self.vtermstore[::1], 'm:', linewidth=3, label = r"$V_{term}$", solid_capstyle='round')
        ##plt.title('Velocity vs Time')
        #plt.xticks()
        #plt.yticks()
        #plt.xlabel('Time (ns)')
        #plt.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
        #plt.legend(frameon=False)
        #plt.ylabel(r'Velocity ($\frac{m}{s}$)')
        #plt.tight_layout()
# 
        #plt.figure(figsize=(11, 17))
        #plot the acceleration
        ax3 = plt.subplot(3, 2, 3)
        if data == True : ax3.plot(texp[::sk], acceleration_estimate[::sk], 'd', markersize = 10, label = "Mumax+", color = "darkorange")  # Convert to nm
        ax3.plot(np.array(self.time[::5])*1e9, self.accelerations[::5], '-', color = "teal", label = "Model",zorder = 0)
        #plt.plot(texp[::2], acceleration_estimate[::2], 'ro', markersize = 4)  # Convert to nm
        #plt.title('Acceleration vs Time')
        ax3.set_xlabel('Time (ns)')
        ax3.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
        ax3.legend(frameon=False, loc = "lower left", bbox_to_anchor=(0.1, 0.0))
        ax3.set_ylabel(r'Acceleration ($\frac{m}{s^2}$)')
# 
        ax_inset2 = inset_axes(ax3, width="40%", height="40%", loc='center right', borderpad = 1.3,   bbox_to_anchor=(-0.025, 0.1, 1, 1),bbox_transform=ax3.transAxes,)
        if data == True:
            ax_inset2.plot(texp[::sk], acceleration_estimate[::sk], 'd', markersize=6, color='darkorange')
        ax_inset2.plot(np.array(self.time[::5])*1e9, self.accelerations[::5], '-', linewidth=2, color='teal',zorder = 0)
# 
        ax_inset2.set_ylim(-0.5e11,0.02e11)
        ax_inset2.set_xlim(0,1)  # keep the same x range as main plot
        ax_inset2.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
# 
        mark_inset(ax3, ax_inset2, loc1=1, loc2=2, fc="none", ec="0.5", lw=1.2)
# 
        plt.tight_layout()
# 
        plt.subplots_adjust(wspace=0.25)
        # 
        axf = plt.subplot(3,2,4)
        axf.plot(np.array(self.time[::1])*1e9, np.array(self.Fformstore[::1])*1e15, color = 'teal',  label ='Magnetoelastic')
        axf.plot(np.array(self.time[::1])*1e9, -np.array(self.Fvstore[::1])*1e15, color = "#D50079", label =r'$-\dot{\chi}$ Damping')
        axf.plot(np.array(self.time[::1])*1e9, np.array(self.Fdeltastore[::1])*1e15, color = "#2D65DF", label =r'$\dot{\Delta}$ Force')
        #plt.title('Forces')
        axf.legend(frameon=False, loc = "lower right", bbox_to_anchor=(0.97, 0.05))
        axf.set_xlabel('Time (ns)')
        axf.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
        axf.set_ylabel(r'Force ($fN$)')
# 
        # --- Inset ---
        ax_inset = inset_axes(
            axf,
            width="40%",        # width of inset
            height="40%",       # height of inset
            loc='upper right',  # anchor point
            bbox_to_anchor=(0, 0, 1, 1),  # relative bbox (full axes)
            bbox_transform=axf.transAxes,
            borderpad=1.4         # padding from the edge; increase to move inset down
        )        
        # Plot the same curves in the inset
        ax_inset.plot(np.array(self.time[::1])*1e9, np.array(self.Fformstore[::1])*1e15, color='teal')
        ax_inset.plot(np.array(self.time[::1])*1e9, -np.array(self.Fvstore[::1])*1e15, color="#D50079")
        ax_inset.plot(np.array(self.time[::1])*1e9, np.array(self.Fdeltastore[::1])*1e15, color="#2D65DF")
# 
        # Set the zoomed y-range
        ax_inset.set_ylim(-3.46, -3.43)
# 
        # Keep the same x-range as the main plot
        ax_inset.set_xlim(axf.get_xlim())
# 
        # Optional: grid for inset
        ax_inset.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
# 
        # Draw rectangle and connectors on main plot
        mark_inset(axf, ax_inset, loc1=3, loc2=4, fc="none", ec="0.5", lw=1.2)
# 
        plt.tight_layout()
# 
        #plt.figure(figsize=(11,5))
        ax1 = plt.subplot(3,2,6)
        # Plot domain wall width on primary y-axis
        ax1.plot(np.array(self.time[::1])*1e9, np.array(self.Kstore[::1])*1e-3,
                 color = "#2D65DF")
        #ax1.set_title('Domain wall width')
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel(r'K$_{eff}$ ($kJ/m^3$)', color="#2D65DF")
        ax1.tick_params(axis='y', labelcolor="#2D65DF")
        ax1.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
        # Create second y-axis for strain
        ax2 = ax1.twinx()
        ax2.plot(np.array(self.time[::1])*1e9, np.array(self.strainstore[::1])*1e3,
                 color = "teal")
        ax2.set_ylabel(r'Strain $\cdot 1e-3$', color='teal')
        ax2.tick_params(axis='y', labelcolor='teal')
        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', frameon = False)
        ymin = np.min(np.array(self.Kstore[::1])*1e-3) - 0.5
        ymax = np.max(np.array(self.Kstore[::1])*1e-3) + 0.5
        ax1.set_ylim([ymin, ymax])
        strain_min = np.min(np.array(self.strainstore[::1])*1e3)
        strain_max = np.max(np.array(self.strainstore[::1])*1e3)
# 
        # Expand the range by a factor
        center = 0.5 * (strain_max + strain_min)
        half_range = 0.5 * (strain_max - strain_min)
        scale_factor = 3.0  # Increase space 3x
        ax2.set_ylim(center - scale_factor * half_range,
             center + scale_factor * half_range)
        plt.tight_layout()
        #plt.subplot(1,3,1)
        #plt.plot(np.array(self.time[::1])*1e9, np.array(self.deltastore[::1])*1e9, 'b-', linewidth=3, label ='Model')
        #plt.plot(np.array(self.time[::1])*1e9, np.array(self.strainstore[::1])*1e3, 'g-', linewidth=3, label ='Strain [1e-3]')
        #plt.title('Domain wall width')
        #plt.xlabel('Time (ns)', fontsize = 16)
        #plt.xticks(fontsize = 14)
        #plt.yticks(fontsize = 14)
        #plt.grid()
        #plt.legend()
        #plt.ylabel(r'$\Delta$ ($nm$)', fontsize = 16)
        # 
# 
        plt.subplot(3,2,5)
        plt.plot(np.array(self.time[::1])*1e9, np.array(self.deltastore[::1])*1e9, color = 'teal')
        #plt.title('Effective Anisotropy')
        plt.xlabel('Time (ns)')
        plt.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
        plt.ylabel(r'$\Delta$ ($nm$)')
        plt.tight_layout()
        plt.subplots_adjust(right=0.9)
        # 
        plt.figure(figsize=(18,4))
        plt.subplot(1,3,1)
        plt.plot(np.array(self.time[::1])*1e9, np.array(self.angle[::1]), color = 'teal', label ='Azimuthal angle [rad]')
        plt.title('Azimuthal angle')
        plt.xlabel('Time (ns)')
        plt.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
        plt.ylabel(r'$\phi$ (rad)')
        plt.tight_layout()
# 
        plt.subplot(1,3,2)
        plt.plot(np.array(self.time[::1])*1e9, np.array(self.wstore[::1]), color = 'teal', label ='Azimuthal angular frequency [rad/s]')
        plt.title('Azimuthal angular frequency')
        plt.xlabel('Time (ns)')
        plt.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
        plt.ylabel(r'$\dot{\phi}$ (rad/s)')
        plt.tight_layout()
# 
        plt.subplot(1,3,3)
        plt.plot(np.array(self.time[::1])*1e9, np.array(self.paccstore[::1]), color = 'teal', label =r'Azimuthal angular acc (rad/$s^2$)')
        plt.title('Azimuthal angular acceleration')
        plt.xlabel('Time (ns)')
        plt.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
        plt.ylabel(r'$\ddot{\phi}$ (rad/$s^2$)')
        plt.tight_layout()
        # 
        self.plot_fft(np.array(self.angle))
        # 
        plt.subplots_adjust(wspace=0.1)
        plt.show()

def x_v_phase_plot(self):
        plt.plot(np.array(self.positions[::5])*1e6, self.velocities[::5], 'b-', linewidth =2)
        plt.title("Phase space")
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.xlabel(r'Position (\mu m)', fontsize = 18)
        plt.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
        plt.ylabel(r'Velocity ($\frac{m}{s}$)', fontsize = 18)
        plt.show()

def _1D_anim_SAW(self,data=False, save = False, follow = False):
        delta = np.sqrt(2*self.A/(2*self.K))
        x_plot = np.linspace(0, self.L, 5000)  # Length of the wire

        if self.potential == True: e_pot = self._potential_force(self.aa, self.bb, x_plot)[0]

        #From DW position to DW profile
        def theta(Z,Zc,delta):
            return 2*(np.arctan(np.exp((Z-Zc)/delta)))

        skip =100
        if data == True:
            Tk().withdraw()  # Close the root window
            filename = askopenfilename(title="Select the .npy file")
            print(f"Selected file: {filename}")
            DW_pos = np.load(filename)[::skip]
        else:
            DW_pos = self.positions[::skip]
        DW_prof = []
        for i in tqdm(range(len(DW_pos))):
            DW_profx = theta(x_plot, DW_pos[i],self.deltastore[::skip][i])
            DW_prof.append(DW_profx)
        
        phi_angl = self.angle[::skip]

        #1D plot over X
        fig, ax = plt.subplots(figsize=(16,8))
        ax2 = ax.twinx()
        if self.potential == True: ax3 = ax.twinx()
        lamb = 5e-7
        k = 2*np.pi/lamb
        dt = self.timestep
        t = 0
        e0 = self.strain
       
        if re.search(r"SAW",self.exc):
            llabel = fr"SAW, vsaw = {self.vsaw:0.0f} m/s, lambda = {lamb*1e9} nm"
        elif self.exc == 'oscill':
            llabel = fr"frequency = {self.of:0.1e} Hz"
        else:
            llabel = "linear strain"
       
       
        if self.strain_type == 'xz':
            line, = ax2.plot(x_plot*1e6, self._strain(x_plot,t,k,e0,phi=0)/2 * np.ones_like(x_plot) if np.isscalar(self._strain(x_plot,t,k,e0,phi=0)) else self._strain(x_plot,t,k,e0,phi=0)/2, 'k-', linewidth = 2, label = llabel, alpha = 0.7)
            line2, = ax2.plot(x_plot*1e6, -self._strain(x_plot,t,k,e0,phi=0)/2* np.ones_like(x_plot) if np.isscalar(-self._strain(x_plot,t,k,e0,phi=0)) else -self._strain(x_plot,t,k,e0,phi=0)/2, 'r-', linewidth = 2, label = llabel, alpha = 0.7)
        else:
            line, = ax2.plot(x_plot*1e6, self._strain(x_plot,t,k,e0,phi=0) * np.ones_like(x_plot) if np.isscalar(self._strain(x_plot,t,k,e0,phi=0)) else self._strain(x_plot,t,k,e0,phi=0), 'k-', linewidth = 2, label = llabel,  alpha = 0.7)

        linx, = ax.plot(x_plot*1e6,np.sin(DW_prof[0])*np.cos(phi_angl[0]),'b', linewidth =4, label = r'm$_x$')
        linz, = ax.plot(x_plot*1e6,np.cos(DW_prof[0]),'r', linewidth =4, label = r'm$_z$')
        liny, = ax.plot(x_plot*1e6,np.sin(DW_prof[0])*np.sin(phi_angl[0]),'m', linewidth =4, label = r'm$_y$')

        if self.potential == True: linpot, = ax3.plot(x_plot*1e6, e_pot, 'g-', linewidth = 2, label = r'Potential energy')
        #line, = ax2.plot(x_plot*1e6, e0*np.sin(k*(x_plot - self.vsaw*t)), 'k--', linewidth = 4, label = fr"SAW, self.vsaw = {self.vsaw:0.0f} m/s, lambda = {lamb*1e9} nm")
       
        ax2.set_ylabel("Strain [/]", fontsize = 18)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax2.tick_params(axis='both', which='major', labelsize=18)
        ax.ticklabel_format(style = "sci", scilimits= (0,0))
        ax2.ticklabel_format(style = "sci", scilimits= (0,0))
        ax2.set_ylim(-e0*3,e0*3)
        ax.set_ylim(-1.01, 1.01)
        ax.set_xlim(x_plot[0]*1e6, x_plot[-1]*1e6)
        if follow == True:
            ax.set_xlim(DW_pos[0]*1e6 -0.05, DW_pos[0]*1e6 + 0.05)
        ax.set_title("test")
        title_text = ax.set_title("", fontsize = 18)
        ax.set_xlabel("x [um]", fontsize = 18)
        ax.set_ylabel("Magnetization [/]", fontsize = 18)
        ax.legend(loc = "upper left", fontsize = 16)
        ax2.legend(loc = "upper right", fontsize = 16)
        ax.grid(axis='x', which='both', linestyle='--', color='gray', alpha=0.7)
        def update(frame):
            if follow == True:
                ax.set_xlim(DW_pos[frame]*1e6 -0.05, DW_pos[frame]*1e6 + 0.05)
            tt = frame*skip*dt
            if self.strain_type == 'xz':
                line2.set_ydata(-self._strain(x = x_plot,t=tt,k = k,e0 = e0, phi = 0)/2* np.ones_like(x_plot) if np.isscalar(-self._strain(x = x_plot,t=tt,k = k,e0 = e0, phi = 0)) else -self._strain(x = x_plot,t=tt,k = k,e0 = e0, phi = 0)/2)
                line.set_ydata(self._strain(x = x_plot,t=tt,k = k,e0 = e0, phi = 0)/2* np.ones_like(x_plot) if np.isscalar(self._strain(x = x_plot,t=tt,k = k,e0 = e0, phi = 0)) else self._strain(x = x_plot,t=tt,k = k,e0 = e0, phi = 0)/2)
            else :
                line.set_ydata(self._strain(x = x_plot,t=tt,k = k,e0 = e0, phi = 0) * np.ones_like(x_plot) if np.isscalar(self._strain(x = x_plot,t=tt,k = k,e0 = e0, phi = 0)) else self._strain(x = x_plot,t=tt,k = k,e0 = e0, phi = 0))
            linx.set_ydata(np.sin(DW_prof[frame])*np.cos(phi_angl[frame]))
            linz.set_ydata(np.cos(DW_prof[frame]))
            liny.set_ydata(np.sin(DW_prof[frame])*np.sin(phi_angl[frame]))
            #line.set_ydata(e0*np.sin(k*(x_plot - self.vsaw*frame*skip*dt)))
            title_text.set_text(f"t = {int(frame)*dt*skip*1e9:.2e} ns")
            if self.strain_type == 'xz': return linx, linz,liny,line,title_text,line2
            else:
                return linx, linz,liny,line,title_text
            
        ani = animation.FuncAnimation(fig, update, frames=np.shape(DW_prof)[0], blit=False, interval = 5)
        #Save the animation
        if save == True: ani.save('1D_SAW_animation.gif')
        plt.show()

def plot_fft(self,omega, fs, color, label, map):
    if omega is None:
         print("ok!")
         omega = np.array(self.angle)
    else : plt.figure(figsize=(8, 4))
    n = len(omega)                  # Number of samples
    t = np.arange(n) * fs           # Time vector (not strictly needed unless you want to plot time)
    if map == 'x':
         omega = np.cos(omega)
    elif map == 'y':
         omega = np.sin(omega)
    fft_vals = np.fft.fft(omega)    # Compute FFT
    fft_freqs = np.fft.fftfreq(n, fs)  # Frequency bins

    # Only take the positive half of frequencies and normalize
    half_n = n // 2
    freqs = fft_freqs[:half_n]
    fft_magnitude = np.abs(fft_vals[:half_n]) * 2 / n

    # Plot
    plt.plot(freqs, fft_magnitude, '-', linewidth=3, label = label, color = color)
    plt.title("FFT of Azimuthal angle")
    plt.xlabel("Frequency (Hz)", fontsize = 16)
    plt.ylabel("Magnitude", fontsize = 16)
    plt.grid(True)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.xlim(0,5e12)
    plt.tight_layout()