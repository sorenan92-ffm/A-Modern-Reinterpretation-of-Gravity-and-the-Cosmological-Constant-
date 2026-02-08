"""
3D Gravity Simulator - Short and Fixed Version
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Gravity3D:
    def __init__(self):
        self.G = 6.67430e-11  # Gravitational constant
        self.dt = 0.1         # Time step (increased for faster simulation)
        self.softening = 1e-6  # Softening parameter
        
    def create_solar_system(self):
        """Create a simple solar system"""
        # 4 main bodies
        masses = np.array([
            1.989e30,    # Sun
            5.972e24,    # Earth
            1.899e27,    # Jupiter
            5.685e26     # Saturn
        ], dtype=np.float64)
        
        # Initial positions (meters) - scaled down for better visualization
        positions = np.array([
            [0, 0, 0],            # Sun
            [1.5e11, 0, 0],       # Earth
            [7.8e11, 0, 0],       # Jupiter
            [1.4e12, 0, 0]        # Saturn
        ], dtype=np.float64)
        
        # Initial velocities (m/s) - tangential for circular orbits
        velocities = np.array([
            [0, 0, 0],           # Sun
            [0, 29780, 0],       # Earth
            [0, 13070, 0],       # Jupiter
            [0, 9690, 0]         # Saturn
        ], dtype=np.float64)
        
        return masses, positions, velocities
    
    def compute_forces(self, positions, masses):
        """Compute gravitational forces between all bodies"""
        n = len(masses)
        forces = np.zeros_like(positions)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    r_vec = positions[j] - positions[i]
                    r = np.linalg.norm(r_vec) + self.softening
                    force = self.G * masses[i] * masses[j] * r_vec / r**3
                    forces[i] += force
        
        return forces
    
    def simulate(self, steps=100):
        """Run the simulation"""
        masses, positions, velocities = self.create_solar_system()
        
        # Store paths
        paths = []
        
        for t in range(steps):
            # Store current positions
            paths.append(positions.copy())
            
            # Compute gravitational forces
            forces = self.compute_forces(positions, masses)
            
            # Compute accelerations
            accelerations = forces / masses[:, np.newaxis]
            
            # Update velocities and positions
            velocities = velocities + accelerations * self.dt
            positions = positions + velocities * self.dt
            
            if t % 20 == 0:
                print(f"⏳ Step {t}/{steps}")
        
        return np.array(paths), masses
    
    def plot_3d(self, paths):
        """Create 3D visualization of orbits"""
        fig = plt.figure(figsize=(12, 10))
        
        # 1. 3D Orbit Plot
        ax1 = fig.add_subplot(221, projection='3d')
        colors = ['yellow', 'blue', 'orange', 'gold']
        labels = ['Sun', 'Earth', 'Jupiter', 'Saturn']
        
        for i in range(paths.shape[1]):
            # Scale for better visualization
            scaled_path = paths[:, i] / 1e11
            ax1.plot(scaled_path[:, 0], scaled_path[:, 1], scaled_path[:, 2],
                    color=colors[i], label=labels[i], linewidth=1.5, alpha=0.7)
            ax1.scatter(scaled_path[-1, 0], scaled_path[-1, 1], scaled_path[-1, 2],
                       color=colors[i], s=80, edgecolors='black')
        
        ax1.set_xlabel('X (×10¹¹ m)')
        ax1.set_ylabel('Y (×10¹¹ m)')
        ax1.set_zlabel('Z (×10¹¹ m)')
        ax1.set_title('3D Orbits')
        ax1.legend()
        
        # 2. X-Y View
        ax2 = fig.add_subplot(222)
        for i in range(paths.shape[1]):
            scaled_path = paths[:, i] / 1e11
            ax2.plot(scaled_path[:, 0], scaled_path[:, 1], color=colors[i], label=labels[i], alpha=0.7)
            ax2.scatter(scaled_path[-1, 0], scaled_path[-1, 1], color=colors[i], s=50, edgecolors='black')
        ax2.set_xlabel('X (×10¹¹ m)')
        ax2.set_ylabel('Y (×10¹¹ m)')
        ax2.set_title('X-Y View')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. X-Z View
        ax3 = fig.add_subplot(223)
        for i in range(paths.shape[1]):
            scaled_path = paths[:, i] / 1e11
            ax3.plot(scaled_path[:, 0], scaled_path[:, 2], color=colors[i], label=labels[i], alpha=0.7)
            ax3.scatter(scaled_path[-1, 0], scaled_path[-1, 2], color=colors[i], s=50, edgecolors='black')
        ax3.set_xlabel('X (×10¹¹ m)')
        ax3.set_ylabel('Z (×10¹¹ m)')
        ax3.set_title('X-Z View')
        ax3.grid(True, alpha=0.3)
        
        # 4. Distance from Center
        ax4 = fig.add_subplot(224)
        for i in range(1, paths.shape[1]):  # Skip the Sun
            distance = np.sqrt(np.sum(paths[:, i]**2, axis=1)) / 1e11
            ax4.plot(distance, color=colors[i], label=labels[i])
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Distance from Sun (×10¹¹ m)')
        ax4.set_title('Orbital Radius')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Show simulation info
        self.show_info(paths)
    
    def show_info(self, paths):
        """Display simulation information"""
        print("\n📊 Simulation Info:")
        print("=" * 40)
        
        # Calculate orbital parameters
        for i, name in enumerate(['Sun', 'Earth', 'Jupiter', 'Saturn']):
            if i > 0:  # Skip the Sun
                # Calculate total distance traveled
                diffs = np.diff(paths[:, i], axis=0)
                distances = np.sqrt(np.sum(diffs**2, axis=1))
                total_distance = np.sum(distances)
                
                # Calculate average orbital radius
                mean_radius = np.mean(np.sqrt(np.sum(paths[:, i]**2, axis=1)))
                
                print(f"\n{name}:")
                print(f"  Total distance traveled: {total_distance:.2e} m")
                print(f"  Average orbital radius: {mean_radius:.2e} m")
                print(f"  Final position (×10¹¹ m): ({paths[-1, i, 0]/1e11:.2f}, {paths[-1, i, 1]/1e11:.2f}, {paths[-1, i, 2]/1e11:.2f})")
                
                # Calculate approximate number of orbits
                circumference = 2 * np.pi * mean_radius
                if total_distance > 0:
                    num_orbits = total_distance / circumference
                    print(f"  Approx. orbits completed: {num_orbits:.3f}")
                    
                    # Calculate orbital period (approximate)
                    if num_orbits > 0:
                        period = (len(paths) * self.dt) / num_orbits
                        print(f"  Orbital period: {period/86400:.1f} days")

# Run the simulation
if __name__ == "__main__":
    print("=" * 50)
    print("🌌 3D Gravity Simulator")
    print("=" * 50)
    
    # Create simulator
    simulator = Gravity3D()
    
    # Run simulation with fewer steps for speed
    print("\n🚀 Running simulation...")
    paths, masses = simulator.simulate(steps=100)
    
    print("\n✅ Simulation complete!")
    print(f"Time steps: {len(paths)}")
    print(f"Number of bodies: {len(masses)}")
    print(f"Total mass: {masses.sum():.2e} kg")
    
    # Display results
    print("\n🎨 Creating visualizations...")
    simulator.plot_3d(paths)
    
    print("\n🎉 All done!")