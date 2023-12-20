import numpy as np

def impens(b_0, b_target, timelst, **kwargs):
    dy = kwargs.get("dy", 0)

    b_i = 0.5*(b_target+b_0) + 1j*dy
    
    l = len(timelst)     # must be odd so that tau/2 is a time point.
    midpoint = l//2+1
    tau = timelst[-1]
    t_half = timelst[:midpoint]
    
    db_firsthalf = np.full(shape=(midpoint,), fill_value=2/timelst[-1]*(b_i-b_0))
    b_firsthalf = b_0 + db_firsthalf * t_half
    
    db_secondhalf = np.full(shape=(midpoint,), fill_value=2/tau*(b_target-b_i))
    b_secondhalf = b_i + db_secondhalf * t_half
    
    b_trajectory = np.concatenate((b_firsthalf, b_secondhalf[1:]))
    db_trajectory = np.concatenate((db_firsthalf, db_secondhalf[1:]))
    return b_trajectory, db_trajectory