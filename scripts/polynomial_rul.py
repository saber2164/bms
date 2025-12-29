import numpy as np

class PolynomialRULPredictor:
    def __init__(self, poly_order=2, eol_soh=0.8):
        self.poly_order = poly_order
        self.eol_soh = eol_soh

    def predict(self, cycle_data, current_cycle):
        """
        Fit polynomial to data up to current_cycle.
        Solve for SoH = EOL_SOH.
        
        Args:
            cycle_data (pd.DataFrame): DataFrame with 'cycle' and 'soh' columns.
            current_cycle (int): The current cycle number to predict from.
            
        Returns:
            float: Predicted Remaining Useful Life (cycles), or np.nan if prediction fails.
        """
        history = cycle_data[cycle_data['cycle'] <= current_cycle]
        
        if len(history) < 5:
            return np.nan # Not enough data
            
        x = history['cycle'].values
        y = history['soh'].values
        
        # Fit polynomial: soh = ax^2 + bx + c
        try:
            z = np.polyfit(x, y, self.poly_order)
            p = np.poly1d(z)
            
            # Solve p(cycle) = EOL_SOH
            # ax^2 + bx + c - EOL_SOH = 0
            roots = (p - self.eol_soh).roots
            
            # Filter real roots > current_cycle
            real_roots = [r.real for r in roots if np.isreal(r) and r.real > current_cycle]
            
            # Fallback to linear fit if no valid roots for polynomial
            if not real_roots:
                z1 = np.polyfit(x, y, 1)
                p1 = np.poly1d(z1)
                roots1 = (p1 - self.eol_soh).roots
                real_roots = [r.real for r in roots1 if np.isreal(r) and r.real > current_cycle]
                
            if real_roots:
                eol_cycle_pred = min(real_roots)
                return eol_cycle_pred - current_cycle
            else:
                return np.nan
                
        except Exception:
            return np.nan
