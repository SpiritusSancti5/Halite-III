#!/usr/bin/env python3

import os
from collections import defaultdict

import hlt
import model
from hlt import constants
from hlt.positionals import Direction, Position
import random
import logging
from math import sqrt
""" <<<Game Begin>>> """
""" <<<Game Loop>>> """
ship_status = {}
class SVMBot:
    def __init__(self, weights):
        # Get the initial game state
        game = hlt.Game()
        logging.info("Successfully created bot! My Player ID is {}.".format(game.my_id))
        game.ready("SVM-" + os.path.basename(weights))

        # During init phase: initialize the model and compile it
        my_model = model.HaliteModel(weights=weights)

        self.my_model = my_model
        self.game = game

    def run(self):
        # Some minimal state to say when to go home
        go_home = defaultdict(lambda: False)
        while True:
            target_positions = []
            self.game.update_frame()
            player_count = len(self.game.players)
            me = self.game.me  # Here we extract our player metadata from the game state
            game_map = self.game.game_map  # And here we extract the map metadata
            other_players = [p for pid, p in self.game.players.items() if pid != self.game.my_id]

            command_queue = []

            for ship in me.get_ships():  # For each of our ships
                logging.info("Ship {} has {} halite. Pos {}.".format(ship.id, ship.halite_amount, ship.position))
                if ship.id not in ship_status:
                    ship_status[ship.id] = "exploring"

                dist_to_shipyard = sqrt((ship.position.x - me.shipyard.position.x) * \
                    (ship.position.x - me.shipyard.position.x) \
                    + (ship.position.y - me.shipyard.position.y) * (ship.position.y - me.shipyard.position.y))
                the_end = self.game.turn_number > constants.MAX_TURNS - dist_to_shipyard - 10
                if ship_status[ship.id] == "returning":
                    if ship.position == me.shipyard.position and not the_end:
                        ship_status[ship.id] = "exploring"
                    elif dist_to_shipyard == 1 and (me.shipyard.position not in target_positions or \
                    the_end):
                        loc = game_map._get_target_direction(ship.position, me.shipyard.position)
                        target_positions.append(me.shipyard.position)
                        game_map[me.shipyard.position].mark_unsafe(ship)
                        if loc[1] == None:
                            command_queue.append(ship.move( loc[0] ))
                        else:
                            command_queue.append(ship.move( loc[1] ))                
                        continue
                    else:
                        move = game_map.naive_navigate(ship, me.shipyard.position)
                        command_queue.append(ship.move(move))
                        continue
                elif ship.halite_amount >= constants.MAX_HALITE * 0.95 or \
                (the_end and ship.halite_amount >= 255):
                    ship_status[ship.id] = "returning"
                    move = game_map.naive_navigate(ship, me.shipyard.position)
                    command_queue.append(ship.move(move))
                    continue
                    
                # Use machine learning to get a move
                ml_move = self.my_model.predict_move(ship, game_map, me, other_players, self.game.turn_number)
                if ml_move is not None:
                    movement = game_map.get_safe_move(game_map[ship.position],
                                                      game_map[ship.position.directional_offset(ml_move)])
                    if movement is not None:
                        game_map[ship.position.directional_offset(movement)].mark_unsafe(ship)
                        target_positions.append(ship.position.directional_offset(movement))
                        command_queue.append(ship.move(movement))
                        continue
                    
                # default    
#                if game_map[ship.position].halite_amount < 50 or ship.is_full:
                poses = ship.position.get_surrounding_cardinals()
                poses.insert(0,ship.position)
                goodies = [0,0,0,0,0]
                best_idx = 0
                curr_halite = -1000 if ship.position == me.shipyard.position else game_map[ship.position].halite_amount
                most_halite = curr_halite*7/16
                for p in range(5):
                    if game_map[poses[p]].is_occupied or poses[p] in target_positions:
                        continue
                    goodies[p] = game_map[poses[p]].halite_amount
                    if p != 0:
                        goodies[p] /= 4
                    if goodies[p] > most_halite or (random.randint(0,2) > 0 and most_halite == 0):
                        most_halite = goodies[p]
                        best_idx = p

                command_queue.append(ship.move( game_map.naive_navigate(ship, poses[best_idx]) ))
                target_positions.append(poses[best_idx])
                game_map[poses[best_idx]].mark_unsafe(ship)
#                else:
#                    command_queue.append(ship.stay_still())

            # Spawn some more ships
            max_ships = 10
            if player_count == 2:
                for player in range(player_count):
                    if player != self.game.my_id:
                        max_ships = max(max_ships,len(self.game.players[player].get_ships()) )
            else:
                max_ships = 50
                player_max_ships = 0
                for player in range(player_count):
                    if player != self.game.my_id:
                        player_max_ships = max(player_max_ships,len(self.game.players[player].get_ships())+1)
                max_ships = min(player_max_ships,max_ships)
            if len(me.get_ships()) < max_ships and self.game.turn_number <= (255 if player_count==2 else 255) \
            and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
                command_queue.append(me.shipyard.spawn())

            self.game.end_turn(command_queue)
