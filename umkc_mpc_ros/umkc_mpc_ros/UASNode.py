#!/usr/bin/env python


class UAS():
    def __init__(self, id:int, home_position:list, 
                 goal_position:list, **kwargs):
        self.id = id # id of UAS
        self.home_position = home_position
        self.goal_position = goal_position  
        self.velocity = kwargs.get('velocity', 10)
        self.altitude = kwargs.get('altitude', 10)
        self.group_size = kwargs.get('group_size', 1)
        self.wind_condition = kwargs.get('wind_condition', 0)
        
    def fly_to_wps(self, waypoints:list):
        pass

if __name__ == '__main__':
    
    uas = UAS(2)
    print("uas id is", uas.id, uas.velocity)
    
    
        
    
    