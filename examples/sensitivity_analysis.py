"""
Script di esecuzione delle analisi (strumenti chiamati da sensitivity_analysis) e 
generazione dei grafici
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from cnav.mission import Mission
from cnav.sensitivity_analysis_tools import (
    run_linear_sensitivity,
    run_sensitivity_vs_range,
    run_sensitivity_3d_surface,
    run_interactive_sensitivity
)

if __name__ == "__main__":

    """Batteria, Fan"""

    missione_h2 = Mission(range_nmi=1000, cruise_speed_kt=450, propulsor="fan")
    run_linear_sensitivity(missione_h2, "Hydrogen combustion")
    
    range_fc = np.linspace(100, 1500, 20)
    run_sensitivity_vs_range("Hydrogen fuel cell", "propeller", 250, range_fc)

    range_grid = np.linspace(200, 2000, 15)
    speed_grid = np.linspace(350, 500, 15)
    run_sensitivity_3d_surface("Hydrogen combustion", "fan", "gamma_tank", range_grid, speed_grid)

    range_esaf = np.linspace(100, 3000, 20)
    run_interactive_sensitivity("e-SAF combustion", "fan", range_esaf, min_speed=300, max_speed=550)

    plt.show()

    """Batteria, Elica"""

    missione_h2 = Mission(range_nmi=1000, cruise_speed_kt=450, propulsor="fan")
    run_linear_sensitivity(missione_h2, "Hydrogen combustion")
        
    range_fc = np.linspace(100, 1500, 20)
    run_sensitivity_vs_range("Hydrogen fuel cell", "propeller", 250, range_fc)

    range_grid = np.linspace(200, 2000, 15)
    speed_grid = np.linspace(350, 500, 15)
    run_sensitivity_3d_surface("Hydrogen combustion", "fan", "gamma_tank", range_grid, speed_grid)
    
    range_esaf = np.linspace(100, 3000, 20)
    run_interactive_sensitivity("e-SAF combustion", "fan", range_esaf, min_speed=300, max_speed=550)
    
    plt.show()

    """Fuel Cell, Fan"""

    missione_h2 = Mission(range_nmi=1000, cruise_speed_kt=450, propulsor="fan")
    run_linear_sensitivity(missione_h2, "Hydrogen combustion")
        
    range_fc = np.linspace(100, 1500, 20)
    run_sensitivity_vs_range("Hydrogen fuel cell", "propeller", 250, range_fc)

    range_grid = np.linspace(200, 2000, 15)
    speed_grid = np.linspace(350, 500, 15)
    run_sensitivity_3d_surface("Hydrogen combustion", "fan", "gamma_tank", range_grid, speed_grid)
    
    range_esaf = np.linspace(100, 3000, 20)
    run_interactive_sensitivity("e-SAF combustion", "fan", range_esaf, min_speed=300, max_speed=550)
    
    plt.show()

    """Fuel Cell, Elica"""

    missione_h2 = Mission(range_nmi=1000, cruise_speed_kt=450, propulsor="fan")
    run_linear_sensitivity(missione_h2, "Hydrogen combustion")
        
    range_fc = np.linspace(100, 1500, 20)
    run_sensitivity_vs_range("Hydrogen fuel cell", "propeller", 250, range_fc)

    range_grid = np.linspace(200, 2000, 15)
    speed_grid = np.linspace(350, 500, 15)
    run_sensitivity_3d_surface("Hydrogen combustion", "fan", "gamma_tank", range_grid, speed_grid)
    
    range_esaf = np.linspace(100, 3000, 20)
    run_interactive_sensitivity("e-SAF combustion", "fan", range_esaf, min_speed=300, max_speed=550)
    
    plt.show()  

    """Combustione Idrogeno, Fan"""

    missione_h2 = Mission(range_nmi=1000, cruise_speed_kt=450, propulsor="fan")
    run_linear_sensitivity(missione_h2, "Hydrogen combustion")
        
    range_fc = np.linspace(100, 1500, 20)
    run_sensitivity_vs_range("Hydrogen fuel cell", "propeller", 250, range_fc)

    range_grid = np.linspace(200, 2000, 15)
    speed_grid = np.linspace(350, 500, 15)
    run_sensitivity_3d_surface("Hydrogen combustion", "fan", "gamma_tank", range_grid, speed_grid)
    
    range_esaf = np.linspace(100, 3000, 20)
    run_interactive_sensitivity("e-SAF combustion", "fan", range_esaf, min_speed=300, max_speed=550)
    
    plt.show()   

    """Combustione Idrogeno, Elica"""

    missione_h2 = Mission(range_nmi=1000, cruise_speed_kt=450, propulsor="fan")
    run_linear_sensitivity(missione_h2, "Hydrogen combustion")
        
    range_fc = np.linspace(100, 1500, 20)
    run_sensitivity_vs_range("Hydrogen fuel cell", "propeller", 250, range_fc)

    range_grid = np.linspace(200, 2000, 15)
    speed_grid = np.linspace(350, 500, 15)
    run_sensitivity_3d_surface("Hydrogen combustion", "fan", "gamma_tank", range_grid, speed_grid)
    
    range_esaf = np.linspace(100, 3000, 20)
    run_interactive_sensitivity("e-SAF combustion", "fan", range_esaf, min_speed=300, max_speed=550)
    
    plt.show()

    """Combustione e-SAF, Fan"""

    missione_h2 = Mission(range_nmi=1000, cruise_speed_kt=450, propulsor="fan")
    run_linear_sensitivity(missione_h2, "Hydrogen combustion")
        
    range_fc = np.linspace(100, 1500, 20)
    run_sensitivity_vs_range("Hydrogen fuel cell", "propeller", 250, range_fc)

    range_grid = np.linspace(200, 2000, 15)
    speed_grid = np.linspace(350, 500, 15)
    run_sensitivity_3d_surface("Hydrogen combustion", "fan", "gamma_tank", range_grid, speed_grid)
    
    range_esaf = np.linspace(100, 3000, 20)
    run_interactive_sensitivity("e-SAF combustion", "fan", range_esaf, min_speed=300, max_speed=550)
    
    plt.show()   

    """Combustione e-SAF, Elica"""

    missione_h2 = Mission(range_nmi=1000, cruise_speed_kt=450, propulsor="fan")
    run_linear_sensitivity(missione_h2, "Hydrogen combustion")
        
    range_fc = np.linspace(100, 1500, 20)
    run_sensitivity_vs_range("Hydrogen fuel cell", "propeller", 250, range_fc)

    range_grid = np.linspace(200, 2000, 15)
    speed_grid = np.linspace(350, 500, 15)
    run_sensitivity_3d_surface("Hydrogen combustion", "fan", "gamma_tank", range_grid, speed_grid)
    
    range_esaf = np.linspace(100, 3000, 20)
    run_interactive_sensitivity("e-SAF combustion", "fan", range_esaf, min_speed=300, max_speed=550)
    
    plt.show()