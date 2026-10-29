import json
import sys
import urllib.parse
import urllib.request

ENV_FILE = r"C:\Users\AminE\.secrets\trello.env"
BOARD_ID = "6add00820e13a7c7024609e8"


def load_credentials():
    env_vars = {}
    with open(ENV_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env_vars[k.strip()] = v.strip()
    return env_vars["TRELLO_KEY"], env_vars["TRELLO_TOKEN"]


def trello_api(method, endpoint, params=None, data=None):
    key, token = load_credentials()
    if params is None:
        params = {}
    params["key"] = key
    params["token"] = token
    url = f"https://api.trello.com/1/{endpoint}?" + urllib.parse.urlencode(params)
    body = None
    headers = {"Accept": "application/json"}
    if data:
        body = urllib.parse.urlencode(data).encode("utf-8")
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_card(ticket_id):
    cards = trello_api("GET", f"boards/{BOARD_ID}/cards")
    return next((c for c in cards if ticket_id.upper() in c["name"].upper()), None)


def move_card(ticket_id, target_list_name):
    lists = trello_api("GET", f"boards/{BOARD_ID}/lists")
    target_list = next(
        (lst for lst in lists if lst["name"].lower() == target_list_name.lower()), None
    )
    if not target_list:
        valid_lists = [lst["name"] for lst in lists if not lst.get("closed", False)]
        print(f"List '{target_list_name}' not found. Active lists: {valid_lists}")
        return False

    card = get_card(ticket_id)
    if not card:
        print(f"Card for '{ticket_id}' not found.")
        return False

    trello_api("PUT", f"cards/{card['id']}", params={"idList": target_list["id"], "pos": "top"})
    print(f"Moved card {ticket_id} ('{card['name'][:30]}...') to '{target_list['name']}'.")
    return True


def add_comment(ticket_id, comment_text):
    card = get_card(ticket_id)
    if not card:
        print(f"Card for '{ticket_id}' not found.")
        return False

    trello_api("POST", f"cards/{card['id']}/actions/comments", params={"text": comment_text})
    print(f"Added comment to card {ticket_id}.")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python trello_sync.py move <TICKET_ID> <LIST_NAME>")
        print("  python trello_sync.py comment <TICKET_ID> <COMMENT_TEXT>")
        sys.exit(1)

    action = sys.argv[1].lower()
    if action == "move":
        move_card(sys.argv[2], sys.argv[3])
    elif action == "comment":
        add_comment(sys.argv[2], sys.argv[3])
    else:
        # Backward compatibility: python trello_sync.py AFC-501 'Done'
        move_card(sys.argv[1], sys.argv[2])
