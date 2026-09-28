import random
from typing import List, Dict, Tuple, Optional, Set, Any

TEMPLATE_0 = {
    "rooms": ["kitchen", "living_room", "bedroom"],
    "room_connections": {
        "kitchen": ["living_room"],
        "living_room": ["kitchen", "bedroom"],
        "bedroom": ["living_room"],
    },
    "receptacles": {
        "kitchen": ["countertop", "fridge", "microwave", "sink"],
        "living_room": ["coffee_table", "sofa"],
        "bedroom": ["bed", "desk", "drawer"],
    },
    "initial_object_locations": {
        "apple": "fridge",
        "mug": "countertop",
        "book": "coffee_table",
        "lamp": "desk",
        "key": "drawer",
    },
    "closed_receptacles": {"fridge", "microwave", "drawer"},
}

TEMPLATE_1 = {
    "rooms": ["kitchen", "hallway", "office", "bathroom"],
    "room_connections": {
        "kitchen": ["hallway"],
        "hallway": ["kitchen", "office", "bathroom"],
        "office": ["hallway"],
        "bathroom": ["hallway"],
    },
    "receptacles": {
        "kitchen": ["fridge", "stove", "table"],
        "hallway": ["shelf"],
        "office": ["desk", "bookshelf", "chair"],
        "bathroom": ["sink", "cabinet", "towel_rack"],
    },
    "initial_object_locations": {
        "tomato": "fridge",
        "bread": "stove",
        "umbrella": "shelf",
        "pen": "desk",
        "novel": "bookshelf",
        "soap": "sink",
        "towel": "towel_rack",
    },
    "closed_receptacles": {"fridge", "cabinet"},
}

class MiniHouseEnv:
    def __init__(self, template_id: int, target_object: str, target_receptacle: str):
        self.template_id = template_id
        self.template = TEMPLATE_0 if template_id == 0 else TEMPLATE_1
        self.target_object = target_object
        self.target_receptacle = target_receptacle

        # Find initial room of agent (first room in house)
        self.current_room = self.template["rooms"][0]
        # Object locations: object -> receptacle or "inventory"
        self.object_locations = dict(self.template["initial_object_locations"])
        # Closed states
        self.open_receptacles: Set[str] = set()
        # Inventory (holds at most 1 item)
        self.inventory: Optional[str] = None
        self.step_count = 0
        self.max_steps = 15

    def get_receptacle_room(self, receptacle: str) -> Optional[str]:
        for room, recs in self.template["receptacles"].items():
            if receptacle in recs:
                return room
        return None

    def get_valid_actions(self) -> List[str]:
        actions = []
        # Movement actions to connected rooms
        connected = self.template["room_connections"].get(self.current_room, [])
        for room in connected:
            actions.append(f"go to {room}")

        # Local receptacles in this room
        local_recs = self.template["receptacles"].get(self.current_room, [])
        for rec in local_recs:
            # Open closed receptacles
            if rec in self.template["closed_receptacles"] and rec not in self.open_receptacles:
                actions.append(f"open {rec}")

            # Take objects from receptacles in current room if accessible
            is_accessible = (rec not in self.template["closed_receptacles"]) or (rec in self.open_receptacles)
            if is_accessible and self.inventory is None:
                for obj, loc in self.object_locations.items():
                    if loc == rec:
                        actions.append(f"take {obj} from {rec}")

            # Put object in receptacle if holding something
            if is_accessible and self.inventory is not None:
                actions.append(f"put {self.inventory} in {rec}")

        return sorted(actions)

    def get_observation(self) -> str:
        local_recs = self.template["receptacles"].get(self.current_room, [])
        rec_details = []
        for rec in local_recs:
            is_open = rec in self.open_receptacles or rec not in self.template["closed_receptacles"]
            if not is_open:
                rec_details.append(f"{rec} (closed)")
            else:
                items_inside = [obj for obj, loc in self.object_locations.items() if loc == rec]
                if items_inside:
                    rec_details.append(f"{rec} containing {', '.join(items_inside)}")
                else:
                    rec_details.append(f"{rec} (empty)")

        inv_str = f"You are carrying: {self.inventory}." if self.inventory else "Your hands are empty."
        obs = f"You are in the {self.current_room}. Receptacles here: {'; '.join(rec_details)}. {inv_str}"
        return obs

    def step(self, action: str) -> Tuple[str, bool, bool]:
        """
        Execute action.
        Returns: (observation, is_success, is_done)
        """
        self.step_count += 1
        valid_actions = self.get_valid_actions()
        action_clean = action.strip().lower()

        # Check if action matches a valid action
        matched_valid = None
        for va in valid_actions:
            if action_clean == va or action_clean.startswith(va):
                matched_valid = va
                break

        if not matched_valid:
            obs = f"Nothing happens. Action '{action}' is not valid here."
        else:
            if matched_valid.startswith("go to "):
                dest_room = matched_valid[len("go to "):]
                self.current_room = dest_room
                obs = f"You move to the {dest_room}."
            elif matched_valid.startswith("open "):
                rec = matched_valid[len("open "):]
                self.open_receptacles.add(rec)
                obs = f"You open the {rec}."
            elif matched_valid.startswith("take "):
                # take <obj> from <rec>
                parts = matched_valid.split(" from ")
                obj = parts[0][len("take "):]
                self.inventory = obj
                self.object_locations[obj] = "inventory"
                obs = f"You take the {obj}."
            elif matched_valid.startswith("put "):
                # put <obj> in <rec>
                parts = matched_valid.split(" in ")
                rec = parts[1]
                obj = self.inventory
                self.inventory = None
                self.object_locations[obj] = rec
                obs = f"You put the {obj} in the {rec}."
            else:
                obs = "Nothing happens."

        # Check goal condition
        success = (self.object_locations.get(self.target_object) == self.target_receptacle)
        done = success or (self.step_count >= self.max_steps)
        full_obs = f"{obs} {self.get_observation()}"
        return full_obs, success, done

    def render_prompt(self, history: List[Tuple[str, str]]) -> str:
        """Render prompt with task goal, recent history, and valid actions list."""
        prompt = (
            f"You are an agent in a house. Goal: put the {self.target_object} in the {self.target_receptacle}.\n"
            f"Rules: You must choose exactly one action from the list of VALID ACTIONS below. Copy it verbatim.\n\n"
        )
        # Keep last 5 history pairs as specified in paper Section 5.1
        recent_history = history[-5:]
        for act, obs in recent_history:
            prompt += f"Action: {act}\nObservation: {obs}\n\n"

        valid = self.get_valid_actions()
        prompt += f"Current State: {self.get_observation()}\n"
        prompt += "VALID ACTIONS:\n" + "\n".join(f"- {a}" for a in valid) + "\n\n"
        prompt += "Choose your next action (output ONLY the chosen action string):"
        return prompt

def generate_minihouse_tasks(n_tasks: int = 30, seed: int = 0) -> List[Dict[str, Any]]:
    """Generate reproducible MiniHouse tasks matching paper specification."""
    rng = random.Random(seed)
    tasks = []
    task_idx = 0

    while len(tasks) < n_tasks:
        template_id = rng.choice([0, 1])
        template = TEMPLATE_0 if template_id == 0 else TEMPLATE_1
        # Pick an object
        obj = rng.choice(list(template["initial_object_locations"].keys()))
        start_rec = template["initial_object_locations"][obj]
        # Pick all available receptacles across all rooms in the house
        all_recs = []
        for room_recs in template["receptacles"].values():
            all_recs.extend(room_recs)
        # Target receptacle must be different from starting location
        candidate_targets = [r for r in all_recs if r != start_rec]
        target_rec = rng.choice(candidate_targets)

        tasks.append({
            "task_id": f"minihouse_{task_idx}",
            "template_id": template_id,
            "target_object": obj,
            "target_receptacle": target_rec,
            "start_receptacle": start_rec,
        })
        task_idx += 1

    return tasks
