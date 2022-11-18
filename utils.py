from config import *

import json
import subprocess
from time import sleep
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_data(path=USERS_FILE, format="json"):
    ext = path.split(".")[-1]
    if ext == "json":
        if format == "json":
            data = json.load(open(path))
        elif format == "pandas":
            data = pd.read_json(path)
        else:
            data = None
    elif ext == "csv":
        data = pd.read_csv(path)
    else:
        data = None

    return data


def load_ref_target_data():
    return pd.read_csv(REF_TARGET_FILE)


def get_user_from_api(username):
    res = subprocess.run(["gh", "api", f"users/{username}"], capture_output=True)
    return json.loads(res.stdout.decode())


def get_user_follows(username):
    user_following = subprocess.run(
        ["gh", "api", f"/users/{username}/following"], capture_output=True
    )
    user_following = json.loads(user_following.stdout.decode())

    if isinstance(user_following, dict):
        return

    follows = []
    for u in user_following:
        follows.append([username, u["login"]])

    user_followers = subprocess.run(
        ["gh", "api", f"/users/{username}/followers"], capture_output=True
    )
    user_followers = json.loads(user_followers.stdout.decode())

    if isinstance(user_followers, dict):
        return

    for u in user_followers:
        follows.append([u["login"], username])

    return follows


def get_repo_lang_ratios(langs_dict):
    total = sum(langs_dict.values())
    if total == 0:
        return {}

    for key in langs_dict:
        langs_dict[key] = round(langs_dict[key] / total, 3)

    return langs_dict


def merge_repo_lang_ratios(orig_langs_dict, langs_dict):
    for key in langs_dict:
        try:
            orig_langs_dict[key] += langs_dict[key]
        except:
            orig_langs_dict[key] = langs_dict[key]

    return orig_langs_dict


def get_user_repos(username):
    user_repos = subprocess.run(
        [
            "gh",
            "api",
            f"/users/{username}/repos?sort=updated_at&direction=desc&per_page=5",
        ],
        capture_output=True,
    )
    user_repos = json.loads(user_repos.stdout.decode())
    if isinstance(user_repos, dict):
        if "message" in user_repos:
            message = user_repos["message"].lower()
            if "not found" in message:
                print(f"{username} not found")
                return -1
            elif "rate limit" in message:
                return
            else:
                return

    languages = {}
    topics = []
    if len(user_repos) > 0:
        user_repos = sorted(user_repos, key=lambda d: d["updated_at"])
        user_repos = user_repos[-USER_REPOS_MAX:]
        for r in user_repos:
            langs = subprocess.run(
                ["gh", "api", f'/repos/{username}/{r["name"]}/languages'],
                capture_output=True,
            )
            langs = langs.stdout.decode()
            if len(langs) > 0:
                langs = json.loads(langs)
            else:
                return

            if "message" in langs:
                if "rate limit" in langs["message"].lower():
                    return
                else:
                    langs = {r["language"]: 1}

            try:
                langs = get_repo_lang_ratios(langs)
            except:
                return

            languages = merge_repo_lang_ratios(languages, langs)
            topics += r["topics"]

        languages = get_repo_lang_ratios(languages)
        topics = Counter(topics)
        topics = dict(topics)
    return languages, topics


def get_absent_users_from_api():
    try:
        users = load_data(format="pandas")
        users = users.replace({np.nan: None})
        usernames = users["login"].values
        users = users.to_dict(orient="records")
    except:
        users = []
        usernames = []

    ref_users = load_ref_target_data()
    ref_usernames = ref_users["name"].values

    absent_usernames = set(ref_usernames).difference(usernames)
    print(f"Found {len(absent_usernames)} absent users")

    # TODO: Threading
    for i, username in enumerate(tqdm(absent_usernames)):
        api_error = True
        while api_error:
            user = get_user_from_api(username)
            if "message" in user:
                message = user["message"].lower()
                if "rate limit" in message:
                    api_error = True
                    print(
                        f"Rate limit error. Waiting for {RATE_LIMIT_WAIT/60} minutes..."
                    )
                    sleep(RATE_LIMIT_WAIT)
                elif "not found" in message:
                    api_error = False
            else:
                users.append(user)
                if (i + 1) % 100 == 0:
                    json.dump(users, open(USERS_FILE, "w"), default=str)
                api_error = False

    json.dump(users, open(USERS_FILE, "w"), default=str)


def get_user_relations_from_api():
    users = load_data(format="pandas")

    try:
        all_user_relations = load_data(USER_ORIG_RELATIONS_FILE, "pandas")
    except:
        all_user_relations = pd.DataFrame(columns=["following", "follow"])

    absent_usernames = set(users["login"].values).difference(
        all_user_relations["following"].values
    )

    print(f"Found {len(absent_usernames)} absent users")

    # TODO: Threading
    for i, username in enumerate(tqdm(absent_usernames)):
        api_error = True
        while api_error:
            user_relations = get_user_follows(username)
            if user_relations is None:
                print(f"Rate limit error. Waiting for {RATE_LIMIT_WAIT/60} minutes...")
                sleep(RATE_LIMIT_WAIT)
            else:
                user_relations = pd.DataFrame(
                    user_relations, columns=["following", "follow"]
                )
                all_user_relations = pd.concat([all_user_relations, user_relations])
                api_error = False

        if (i + 1) % 100 == 0:
            all_user_relations.to_csv(USER_ORIG_RELATIONS_FILE, index=False)

    all_user_relations.to_csv(USER_ORIG_RELATIONS_FILE, index=False)


def get_user_repos_from_api():
    users = load_data(format="pandas")
    try:
        all_user_repos = load_data(USER_REPOS_FILE)
    except:
        all_user_repos = []

    absent_usernames = set(users["login"].values).difference(
        [r_s["username"] for r_s in all_user_repos]
    )

    # TODO: Threading
    for i, username in enumerate(tqdm(absent_usernames)):
        api_error = True
        while api_error:
            user_repos = get_user_repos(username)
            if user_repos == -1:
                api_error = False
            elif user_repos is None:
                print(
                    f"Rate limit error for {username}. Waiting for {RATE_LIMIT_WAIT/60} minutes..."
                )
                sleep(RATE_LIMIT_WAIT)
            else:
                langs, topics = user_repos
                all_user_repos.append(
                    {"username": username, "languages": langs, "topics": topics}
                )
                api_error = False

        if (i + 1) % 100 == 0:
            json.dump(all_user_repos, open(USER_REPOS_FILE, "w"), default=str)

    json.dump(all_user_repos, open(USER_REPOS_FILE, "w"), default=str)


def convert_to_single_label(row):
    new_row = row.copy()
    ind = np.argwhere(new_row.values)[0][0]
    new_row[ind + 1 :] = 0
    return new_row
