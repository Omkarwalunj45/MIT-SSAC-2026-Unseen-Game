import pandas as pd
df=pd.read_excel('Dataset/Project_dataset.xlsx')
df=df[df['year']==2025]
DF=df

########## BATTING ICR CALCULATION #####################
"""
Part 1
Research-Grade Venue-Adjusted Performance (RVAP) Metric
- Computes expected runs using comparison group's avg runs per ball * (avg balls faced by group if out, else player's balls faced)
- Volume adjustment as sum of per-venue avg(adjusted runs / innings) using unique innings
"""

import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime

# ----------------- Helpers -----------------
def log_progress(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

def get_comparison_group(pos):
    """Get comparison group for RAA and WAA calculations"""
    try:
        pos = int(round(pos))
    except Exception:
        return None
    if pos in [1, 2]:
        return [1, 2, 3]
    elif pos == 3:
        return [1, 2, 3, 4]
    elif pos == 4:
        return [3, 4, 5]
    elif pos == 5:
        return [4, 5, 6]
    elif pos == 6:
        return [5, 6, 7]
    elif pos == 7:
        return [6, 7, 8]
    elif pos == 8:
        return [7, 8, 9]
    else:
        return None

# ----------------- Main RVAP Calculation -----------------
def calculate_player_rvap(df, player_name=None, VOLUME_WEIGHT=1.0, detailed=False):
    """
    Compute RVAP scores for all players or provide detailed step-by-step for single player.
    - Expected runs = (avg runs per ball for comp group at venue) * (avg balls faced by group if out=True, else player's balls)
    - Volume adjustment = sum of per-venue avg(adjusted runs / innings), where adjusted runs = score / venue_factor
    - Derives balls from ball_id if balls column is missing
    - Aggregates out column with max to capture dismissal status
    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns: ['p_match', 'inns', 'bat', 'venue', 'final_batting_position', 'score', 'venue_factor', 'date', 'out']
        Optionally includes 'balls' or 'ball_id' to derive balls faced
    player_name : str or None
        If provided and detailed=True, prints step-by-step for this player (case-insensitive match)
    VOLUME_WEIGHT : float
        Weight applied to volume_adjustment in final RVAP formula
    detailed : bool
        If True and player_name provided, prints detailed breakdown
    Returns
    -------
    pandas.DataFrame
        Per-player RVAP summary (player, total_runs, total_innings, volume_adjustment,
        total_venue_deltas, venues_played, rvap_score, primary_pos, comp_group)
    """
    log_progress("Starting Research-Grade Venue-Adjusted Performance Analysis")

    # Step 1: Validate and aggregate to innings level
    log_progress("Aggregating to innings level...")
    required = {'p_match', 'inns', 'bat', 'venue', 'final_batting_position', 'score', 'venue_factor', 'date', 'out'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in df: {missing}")

    # Check for balls or ball_id
    has_balls = 'balls' in df.columns
    has_ball_id = 'ball_id' in df.columns
    if not has_balls and not has_ball_id:
        raise KeyError("Neither 'balls' nor 'ball_id' found in df. Cannot compute expected runs.")

    # Validate out column
    if not df['out'].isin([True, False, 0, 1, np.nan]).all():
        log_progress("Warning: 'out' column contains non-boolean values. Attempting to coerce to boolean.")
        df['out'] = df['out'].replace({1: True, 0: False, np.nan: False}).astype(bool)

    agg_dict = {
        'score': 'sum',
        'venue_factor': 'first',
        'date': 'first',
        'out': 'max'  # Take max to capture any True (dismissed) in innings
    }
    if has_balls:
        agg_dict['balls'] = 'sum'
    if has_ball_id:
        agg_dict['ball_id'] = 'nunique'  # Count unique ball_id as balls faced

    innings_df = df.groupby(['p_match', 'inns', 'bat', 'venue', 'final_batting_position'], as_index=False).agg(agg_dict).reset_index(drop=True)
    innings_df.columns = ['match_id', 'innings_num', 'player', 'venue', 'batting_pos', 'score', 'venue_factor', 'date', 'out'] + \
                         (['balls'] if has_balls else []) + (['ball_id'] if has_ball_id else [])

    # Derive balls from ball_id if balls is missing
    if not has_balls and has_ball_id:
        innings_df['balls'] = innings_df['ball_id']
        innings_df = innings_df.drop(columns=['ball_id'], errors='ignore')
        log_progress("Derived 'balls' from unique 'ball_id' counts per innings")
    elif has_balls and has_ball_id:
        log_progress("Using 'balls' column; ignoring 'ball_id'")

    # Check for duplicate innings
    duplicates = innings_df.duplicated(subset=['match_id', 'innings_num', 'player']).sum()
    if duplicates > 0:
        log_progress(f"Warning: Found {duplicates} duplicate innings (same p_match, inns, bat). Dropping duplicates.")
        innings_df = innings_df.drop_duplicates(subset=['match_id', 'innings_num', 'player'])

    # Normalize whitespace in player names
    innings_df['player'] = innings_df['player'].astype(str).str.strip()

    # Ensure numeric types and handle out column
    innings_df['score'] = pd.to_numeric(innings_df['score'], errors='coerce')
    innings_df['venue_factor'] = pd.to_numeric(innings_df['venue_factor'], errors='coerce').replace(0, np.nan)
    innings_df['balls'] = pd.to_numeric(innings_df['balls'], errors='coerce').fillna(0)
    innings_df['batting_pos'] = pd.to_numeric(innings_df['batting_pos'], errors='coerce')
    innings_df['out'] = innings_df['out'].astype(bool)

    # Debug: Check not-out counts for target player
    if player_name:
        player_rows_temp = innings_df[innings_df['player'].str.lower() == player_name.lower()]
        not_out_count = len(player_rows_temp[player_rows_temp['out'] == False])
        log_progress(f"Debug: {player_name} has {not_out_count} not-out innings (out=False) in dataset")

    # Step 2: Compute comparison group stats (avg runs per ball and avg balls per innings per venue)
    log_progress("Computing comparison group stats...")
    innings_df['comp_group'] = innings_df['batting_pos'].apply(get_comparison_group)
    venue_group_stats = innings_df.groupby(['venue', 'batting_pos']).agg({
        'score': 'sum',
        'balls': 'sum',
        'match_id': 'nunique',  # Count unique matches per venue, batting_pos
        'innings_num': 'count'  # Count innings
    }).reset_index()
    venue_group_stats['runs_per_ball'] = venue_group_stats['score'] / venue_group_stats['balls'].replace(0, np.nan)
    venue_group_stats['avg_balls_per_innings'] = venue_group_stats['balls'] / venue_group_stats['innings_num']

    # Map comp_group stats to each innings
    comp_group_stats = {}
    for venue in innings_df['venue'].unique():
        for pos in innings_df['batting_pos'].unique():
            comp_group = get_comparison_group(pos)
            if comp_group:
                group_data = venue_group_stats[(venue_group_stats['venue'] == venue) & (venue_group_stats['batting_pos'].isin(comp_group))]
                runs_per_ball = group_data['score'].sum() / group_data['balls'].sum() if group_data['balls'].sum() > 0 else 0
                avg_balls = group_data['balls'].sum() / group_data['innings_num'].sum() if group_data['innings_num'].sum() > 0 else 0
                comp_group_stats[(venue, pos)] = {'runs_per_ball': runs_per_ball, 'avg_balls': avg_balls}
                if runs_per_ball == 0 or avg_balls == 0:
                    log_progress(f"Warning: Zero runs per ball or avg balls for venue {venue}, batting_pos {pos}, comp_group {comp_group}")

    def get_expected_runs(row):
        pos = row['batting_pos']
        venue = row['venue']
        player_balls = row['balls']
        out_status = row['out']
        key = (venue, pos)
        stats = comp_group_stats.get(key, {'runs_per_ball': 0, 'avg_balls': 0})
        balls_to_use = player_balls if not out_status else stats['avg_balls']
        return stats['runs_per_ball'] * balls_to_use

    innings_df['expected_runs'] = innings_df.apply(get_expected_runs, axis=1)
    innings_df['venue_delta'] = innings_df['score'] - innings_df['expected_runs']
    innings_df['cumulative_delta'] = innings_df.groupby('player')['venue_delta'].cumsum()
    innings_df['adjusted_runs'] = innings_df['score'] / innings_df['venue_factor'].replace(np.nan, 1.0)

    # Step 3: Prepare list of players to process
    players = [player_name] if player_name else innings_df['player'].unique()
    if player_name:
        all_players = innings_df['player'].unique()
        if player_name not in all_players:
            lower_map = {p.casefold(): p for p in all_players}
            if player_name.casefold() in lower_map:
                players = [lower_map[player_name.casefold()]]
            else:
                matches = [p for p in all_players if player_name.casefold() in p.casefold()]
                if matches:
                    players = [matches[0]]
                else:
                    print(f"[WARN] Player '{player_name}' not found (case-insensitive). No detailed print will be shown.")
                    players = []
    log_progress(f"Found {len(innings_df['player'].unique())} distinct players; processing {len(players)} target(s).")

    # Step 4: Player-level loop and aggregation
    player_rvap_data = {}
    detailed_shown = False

    for player in players:
        player_rows = innings_df[innings_df['player'] == player].copy()
        if player_rows.empty:
            continue

        # Sort by date for chronological output
        player_rows = player_rows.sort_values(['date', 'match_id', 'innings_num']).reset_index(drop=True)

        # Summary numbers
        total_innings = len(player_rows)
        total_runs = player_rows['score'].sum()
        total_inn = player_rows['match_id'].nunique()
        total_venue_deltas = player_rows['venue_delta'].sum()

        # Volume adjustment: sum of per-venue avg(adjusted runs / innings)
        venue_summary = player_rows.groupby('venue').agg({
            'adjusted_runs': 'sum',
            'match_id': 'nunique',
            'innings_num': 'count'  # Count unique innings per venue
        }).rename(columns={'match_id': 'unique_matches', 'innings_num': 'n_innings'})
        venue_summary['avg_adjusted_runs_per_innings'] = venue_summary['adjusted_runs'] / venue_summary['n_innings']

        # Debug: Log venue summary for target player
        if player_name and player.lower() == player_name.lower():
            log_progress(f"Debug: Venue summary for {player}:")
            print(venue_summary[['n_innings', 'unique_matches', 'adjusted_runs', 'avg_adjusted_runs_per_innings']].to_string(index=True))

        volume_adjustment = (venue_summary['avg_adjusted_runs_per_innings'].sum()/total_inn)

        venues_played = player_rows['venue'].nunique()
        primary_pos = Counter(player_rows['batting_pos'].dropna()).most_common(1)[0][0] if player_rows['batting_pos'].notna().any() else None
        comp_group = get_comparison_group(primary_pos) if primary_pos else None
        rvap_score = total_venue_deltas + (VOLUME_WEIGHT * volume_adjustment)

        # Out/Not-out summary
        out_count = len(player_rows[player_rows['out'] == True])
        not_out_count = len(player_rows[player_rows['out'] == False])

        # Save player data
        player_rvap_data[player] = {
            'total_runs': total_runs,
            'total_innings': total_innings,
            'volume_adjustment': volume_adjustment,
            'total_venue_deltas': total_venue_deltas,
            'venues_played': venues_played,
            'rvap_score': rvap_score,
            'primary_pos': primary_pos,
            'comp_group': comp_group
        }

        # Detailed output for specified player
        if detailed and not detailed_shown and player_name and player.lower() == players[0].lower():
            detailed_shown = True
            print("\n" + "="*90)
            print(f"DETAILED RVAP CALCULATION FOR: {player}")
            print("="*90)
            print(f"Player summary:")
            print(f"  Total innings : {total_innings}")
            print(f"  Out innings   : {out_count}")
            print(f"  Not out innings: {not_out_count}")
            print(f"  Total runs    : {total_runs:.2f}")
            print(f"  Volume adj    : {volume_adjustment:.2f}")
            print(f"  Venue deltas  : {total_venue_deltas:.2f}")
            print(f"  Venues played : {venues_played}")
            print(f"  Primary pos   : {primary_pos}  |  Comparison group: {comp_group}")
            print("-"*90)

            # Per-innings breakdown
            display_cols = ['match_id', 'innings_num', 'date', 'venue', 'batting_pos', 'score', 'balls', 'out', 'venue_factor', 'expected_runs', 'venue_delta', 'cumulative_delta', 'adjusted_runs']
            pr = player_rows.copy()
            pr['score'] = pr['score'].round(2)
            pr['balls'] = pr['balls'].round(0)
            pr['venue_factor'] = pr['venue_factor'].round(4)
            pr['expected_runs'] = pr['expected_runs'].round(3)
            pr['venue_delta'] = pr['venue_delta'].round(3)
            pr['cumulative_delta'] = pr['cumulative_delta'].round(3)
            pr['adjusted_runs'] = pr['adjusted_runs'].round(3)
            print("Per-innings breakdown (chronological):")
            print(pr[display_cols].to_string(index=False))
            print("-"*90)

            # Per-venue summary
            venue_summary['score'] = player_rows.groupby('venue')['score'].sum()
            venue_summary['expected_runs'] = player_rows.groupby('venue')['expected_runs'].sum()
            venue_summary['venue_delta'] = player_rows.groupby('venue')['venue_delta'].sum()
            venue_summary['avg_runs_per_innings'] = (venue_summary['score'] / venue_summary['n_innings']).round(3)
            venue_summary['avg_expected_per_innings'] = (venue_summary['expected_runs'] / venue_summary['n_innings']).round(3)
            venue_summary['delta_per_innings'] = (venue_summary['venue_delta'] / venue_summary['n_innings']).round(3)
            print("Per-venue summary:")
            print(venue_summary[['n_innings', 'unique_matches', 'score', 'expected_runs', 'venue_delta', 'avg_runs_per_innings', 'avg_expected_per_innings', 'delta_per_innings', 'adjusted_runs', 'avg_adjusted_runs_per_innings']].to_string(index=False))
            print("-"*90)

            # RVAP formula explanation
            print("COMPONENTS -> FORMULAS & NUMBERS")
            print(f"  1) Venue-specific skill delta (sum over innings):")
            print(f"       venue_delta_i = score_i - (avg_runs_per_ball_comp_group × (player_balls_i if not out else avg_balls_comp_group))")
            print(f"     Sum(venue_delta_i) = {total_venue_deltas:.3f}")
            print()
            print(f"  2) Volume adjustment (sum of per-venue avg adjusted runs):")
            print(f"       adjusted_runs = score_i / venue_factor_i")
            print(f"       avg_adjusted_runs_per_innings = Σ(adjusted_runs) / n_innings (per venue)")
            print(f"       volume_adjustment = Σ(avg_adjusted_runs_per_innings) = {volume_adjustment:.3f}")
            print()
            print(f"  FINAL RVAP:")
            print(f"    rvap_score = total_venue_deltas + (VOLUME_WEIGHT × volume_adjustment)")
            print(f"    Using VOLUME_WEIGHT = {VOLUME_WEIGHT}")
            print(f"    Calculation: {total_venue_deltas:.3f} + ({VOLUME_WEIGHT:.3f} × {volume_adjustment:.3f}) = {rvap_score:.3f}")
            print("="*90)

    # Convert to DataFrame
    result_df = pd.DataFrame.from_dict(player_rvap_data, orient='index').reset_index().rename(columns={'index': 'player'})
    numeric_cols = ['total_runs', 'total_innings', 'volume_adjustment', 'total_venue_deltas', 'venues_played', 'rvap_score']
    for c in numeric_cols:
        result_df[c] = pd.to_numeric(result_df[c], errors='coerce')

    return result_df

# ----------------- Run Example -----------------
if __name__ == "__main__":
    # Assumes df already loaded
    log_progress("Running RVAP for dataset (summary)...")
    results_df = calculate_player_rvap(df, VOLUME_WEIGHT=1.0, detailed=False)
    print("\nTOP 10 PLAYERS BY RVAP SCORE")
    print(results_df.sort_values('rvap_score', ascending=False).head(10))

    # Detailed step-by-step for a single player
    target = "Virat Kohli"
    single_player_df = calculate_player_rvap(df, player_name=target, VOLUME_WEIGHT=1.0, detailed=True)


################################### PART 2 ######################################

import pandas as pd
import numpy as np

# params
K = 5.0   # shrinkage hyperparameter; tune as needed
group_col = 'group'   # grouping used for mu/percentile recomputation (change if needed)
n_col = 'num_matches' # sample size column

# assume player_group_stats is already loaded
df = player_group_stats.copy()

# 1) rename original columns to *_initial
if 'avg_wpa' in df.columns:
    df = df.rename(columns={'avg_wpa': 'avg_wpa_initial'})
else:
    df['avg_wpa_initial'] = np.nan

if 'percentile' in df.columns:
    df = df.rename(columns={'percentile': 'percentile_initial'})
else:
    df['percentile_initial'] = np.nan

# Ensure numeric types
df['avg_wpa_initial'] = pd.to_numeric(df['avg_wpa_initial'], errors='coerce')
df[n_col] = pd.to_numeric(df[n_col], errors='coerce').fillna(0).astype(float)

# 2) compute group means (mu) using available avg_wpa_initial values
group_mu = df.groupby(group_col)['avg_wpa_initial'].mean().rename('mu_group')
df = df.merge(group_mu, left_on=group_col, right_index=True, how='left')

# 3) apply shrinkage: alpha = n / (n + k)
#    avg_wpa_new = alpha * x + (1 - alpha) * mu_group
def shrink_row(row, k=K):
    x = row['avg_wpa_initial']
    mu = row['mu_group']
    n = row[n_col]
    if pd.isna(mu):
        # no group mean available -> fallback to global mean of avg_wpa_initial
        mu = df['avg_wpa_initial'].mean(skipna=True)
    if pd.isna(x):
        # if player's x is NaN, we cannot compute a meaningful shrink toward mu for that player
        # keep result NaN (user can decide to fill with mu if desired)
        return np.nan
    if (n is None) or (n <= 0) or pd.isna(n):
        alpha = 0.0
    else:
        alpha = float(n) / (float(n) + float(k))
    return alpha * x + (1.0 - alpha) * mu

df['avg_wpa'] = df.apply(shrink_row, axis=1)

# 4) compute new percentile from avg_wpa within the same group (0..100, min->0, max->100)
def compute_percentile_groupwise(sub):
    """
    sub: DataFrame for one group
    returns Series of percentiles aligned with sub.index
    Uses (rank - 1)/(n-1)*100 mapping; preserves NaN.
    """
    s = sub['avg_wpa'].astype(float)
    non_na = s.dropna()
    n = len(non_na)
    if n == 0:
        return pd.Series([np.nan] * len(s), index=sub.index)
    ranks = s.rank(method='average', na_option='keep')
    if n == 1:
        pct = ranks.where(ranks.isna(), 100.0)
    else:
        pct = (ranks - 1.0) / (n - 1.0) * 100.0
    pct = pct.where(~s.isna(), np.nan)
    return pct

df['percentile'] = df.groupby(group_col, group_keys=False).apply(lambda g: compute_percentile_groupwise(g))

# 5) drop helper column mu_group if you don't want it
df = df.drop(columns=['mu_group'])

# 6) informative prints / quick-check
print("Shrinkage applied with K =", K)
print("Columns now:", df.columns.tolist())
print("\nSample rows:")
print(df[[group_col, 'avg_wpa_initial', n_col, 'avg_wpa', 'percentile_initial', 'percentile']].head(10).to_string(index=False))

# If you want to assign back to the original variable:
player_group_stats_norm = df
player_group_stats = player_group_stats_norm
df=DF  # new dataframe with normalized avg_wpa and recomputed percentile


################################### PART 3 ######################################



import pandas as pd
import numpy as np

# params
K = 5.0   # shrinkage hyperparameter; tune as needed
group_col = 'group'   # grouping used for mu/percentile recomputation (change if needed)
n_col = 'num_matches' # sample size column

# assume player_group_stats is already loaded
df = player_group_stats.copy()

# 1) rename original columns to *_initial
if 'avg_wpa' in df.columns:
    df = df.rename(columns={'avg_wpa': 'avg_wpa_initial'})
else:
    df['avg_wpa_initial'] = np.nan

if 'percentile' in df.columns:
    df = df.rename(columns={'percentile': 'percentile_initial'})
else:
    df['percentile_initial'] = np.nan

# Ensure numeric types
df['avg_wpa_initial'] = pd.to_numeric(df['avg_wpa_initial'], errors='coerce')
df[n_col] = pd.to_numeric(df[n_col], errors='coerce').fillna(0).astype(float)

# 2) compute group means (mu) using available avg_wpa_initial values
group_mu = df.groupby(group_col)['avg_wpa_initial'].mean().rename('mu_group')
df = df.merge(group_mu, left_on=group_col, right_index=True, how='left')

# 3) apply shrinkage: alpha = n / (n + k)
#    avg_wpa_new = alpha * x + (1 - alpha) * mu_group
def shrink_row(row, k=K):
    x = row['avg_wpa_initial']
    mu = row['mu_group']
    n = row[n_col]
    if pd.isna(mu):
        # no group mean available -> fallback to global mean of avg_wpa_initial
        mu = df['avg_wpa_initial'].mean(skipna=True)
    if pd.isna(x):
        # if player's x is NaN, we cannot compute a meaningful shrink toward mu for that player
        # keep result NaN (user can decide to fill with mu if desired)
        return np.nan
    if (n is None) or (n <= 0) or pd.isna(n):
        alpha = 0.0
    else:
        alpha = float(n) / (float(n) + float(k))
    return alpha * x + (1.0 - alpha) * mu

df['avg_wpa'] = df.apply(shrink_row, axis=1)

# 4) compute new percentile from avg_wpa within the same group (0..100, min->0, max->100)
def compute_percentile_groupwise(sub):
    """
    sub: DataFrame for one group
    returns Series of percentiles aligned with sub.index
    Uses (rank - 1)/(n-1)*100 mapping; preserves NaN.
    """
    s = sub['avg_wpa'].astype(float)
    non_na = s.dropna()
    n = len(non_na)
    if n == 0:
        return pd.Series([np.nan] * len(s), index=sub.index)
    ranks = s.rank(method='average', na_option='keep')
    if n == 1:
        pct = ranks.where(ranks.isna(), 100.0)
    else:
        pct = (ranks - 1.0) / (n - 1.0) * 100.0
    pct = pct.where(~s.isna(), np.nan)
    return pct

df['percentile'] = df.groupby(group_col, group_keys=False).apply(lambda g: compute_percentile_groupwise(g))

# 5) drop helper column mu_group if you don't want it
df = df.drop(columns=['mu_group'])

# 6) informative prints / quick-check
print("Shrinkage applied with K =", K)
print("Columns now:", df.columns.tolist())
print("\nSample rows:")
print(df[[group_col, 'avg_wpa_initial', n_col, 'avg_wpa', 'percentile_initial', 'percentile']].head(10).to_string(index=False))

# If you want to assign back to the original variable:
player_group_stats_norm = df
player_group_stats = player_group_stats_norm
df=DF  # new dataframe with normalized avg_wpa and recomputed percentile


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# Assuming df has the required columns
df = df.copy()

# Filter players with at least 3 innings
innings_count = df[['bat', 'p_match', 'inns']].drop_duplicates().groupby('bat').size().reset_index(name='innings')
valid_players = innings_count[innings_count['innings'] >= 3]['bat']
df = df[df['bat'].isin(valid_players)]

# Define position groups
def get_group(pos):
    if pos in [1, 2]:
        return 'Openers'
    elif pos in [3, 4]:
        return 'Top Order'
    elif pos in [5, 6]:
        return 'Middle Order'
    elif pos in [7, 8]:
        return 'Lower Order'
    return 'Other'

df['group'] = df['final_batting_position'].apply(get_group)

# Handle edge cases
epsilon = 1e-6
df['score'] = df['score'].fillna(0)
df['out'] = df['out'].fillna(False)
df['noball'] = df['noball'].fillna(0)
df['wide'] = df['wide'].fillna(0)
df['byes'] = df['byes'].fillna(0)
df['legbyes'] = df['legbyes'].fillna(0)

# Aggregate metrics per player per group
player_stats = df.groupby(['bat', 'group']).agg({
    'score': 'sum',
    'ball_id': 'count',
    'out': 'sum',
    'p_match': 'nunique'
}).reset_index().rename(columns={'p_match': 'match_count'})

player_stats['SR'] = (player_stats['score'] / (player_stats['ball_id'] + epsilon)) * 100

# Boundary percentage
boundaries = df[df['score'] >= 4].groupby(['bat', 'group'])['ball_id'].count().reset_index(name='boundary_count')
player_stats = player_stats.merge(boundaries, on=['bat', 'group'], how='left')
player_stats['boundary_count'] = player_stats['boundary_count'].fillna(0)
player_stats['Bdry%'] = (player_stats['boundary_count'] / (player_stats['ball_id'] + epsilon)) * 100

# Dot percentage
dots = df[(df['score'] == 0) & (df['noball'] == 0) & (df['wide'] == 0) & (df['byes'] == 0) & (df['legbyes'] == 0)].groupby(['bat', 'group'])['ball_id'].count().reset_index(name='dot_count')
player_stats = player_stats.merge(dots, on=['bat', 'group'], how='left')
player_stats['dot_count'] = player_stats['dot_count'].fillna(0)
player_stats['dot_percentage'] = (player_stats['dot_count'] / (player_stats['ball_id'] + epsilon)) * 100

# Batting average
player_stats['AVG'] = player_stats['score'] / (player_stats['out'] + epsilon)

# Non-boundary strike rate
non_boundary = df[df['score'] < 4].groupby(['bat', 'group']).agg({
    'score': 'sum',
    'ball_id': 'count'
}).reset_index()
non_boundary['nbdry_sr'] = (non_boundary['score'] / (non_boundary['ball_id'] + epsilon)) * 100
player_stats = player_stats.merge(non_boundary[['bat', 'group', 'nbdry_sr']], on=['bat', 'group'], how='left')
player_stats['nbdry_sr'] = player_stats['nbdry_sr'].fillna(0)

# Balls per dismissal
player_stats['BPD'] = player_stats['ball_id'] / (player_stats['out'] + epsilon)

# Balls per boundary
player_stats['BPB'] = player_stats['ball_id'] / (player_stats['boundary_count'] + epsilon)

# 30+ scores
innings_runs = df.groupby(['bat', 'group', 'p_match', 'inns'])['score'].sum().reset_index(name='runs')
thirties = innings_runs[innings_runs['runs'] >= 30].groupby(['bat', 'group'])['p_match'].count().reset_index(name='30s')
player_stats = player_stats.merge(thirties, on=['bat', 'group'], how='left')
player_stats['30s'] = player_stats['30s'].fillna(0)

# Median runs per innings
median_runs = innings_runs.groupby(['bat', 'group'])['runs'].median().reset_index(name='median')
player_stats = player_stats.merge(median_runs, on=['bat', 'group'], how='left')
player_stats['median'] = player_stats['median'].fillna(0)

# Consistency (inverse coefficient of variation of runs per innings)
consistency = innings_runs.groupby(['bat', 'group'])['runs'].apply(
    lambda x: x.mean() / (x.std() + epsilon) if x.std() != 0 else 1.0
).reset_index(name='consistency')
player_stats = player_stats.merge(consistency, on=['bat', 'group'], how='left')

# Ratios
player_stats['bdry_dot_ratio'] = player_stats['Bdry%'] / (player_stats['dot_percentage'] + epsilon)
player_stats['bpd_bpb_ratio'] = player_stats['BPD'] / (player_stats['BPB'] + epsilon)

# Calculate percentiles within each group
metrics = ['SR', 'bdry_dot_ratio', 'AVG', 'median', 'nbdry_sr', 'bpd_bpb_ratio', '30s', 'consistency']
for metric in metrics:
    player_stats[f'percentile_{metric}'] = player_stats.groupby('group')[metric].rank(pct=True) * 100

# PCA-based weights for each group
weights_dict = {}
for group in player_stats['group'].unique():
    if group == 'Other':
        continue  # Skip Other group
    group_data = player_stats[player_stats['group'] == group]
    X = group_data[[f'percentile_{metric}' for metric in metrics]]
    if len(group_data) > len(metrics):  # Ensure enough data for PCA
        pca = PCA(n_components=1)
        pca.fit(X)
        weights = pca.components_[0] / np.sum(np.abs(pca.components_[0]))
    else:
        weights = np.array([1/len(metrics)] * len(metrics))  # Equal weights if PCA not feasible
    weights_dict[group] = dict(zip(metrics, weights))
    print(f"PCA Weights for {group}:")
    for metric, weight in weights_dict[group].items():
        print(f"  {metric}: {weight:.3f}")

# Compute ICR
player_stats['ICR'] = 0.0  # Initialize ICR column as float
for group in weights_dict:
    mask = player_stats['group'] == group
    weights = weights_dict[group]
    icr = sum(weights[metric] * player_stats.loc[mask, f'percentile_{metric}'] for metric in metrics)
    player_stats.loc[mask, 'ICR'] = icr * 100  # Scale to [0, 100]

# Filter out Other group
player_stats = player_stats[player_stats['group'] != 'Other']

# Compute ICR percentiles within each group
player_stats['ICR_percentile'] = player_stats.groupby('group')['ICR'].rank(pct=True) * 100

# Sort by group and ICR percentile (descending)
player_stats = player_stats.sort_values(['group', 'ICR_percentile'], ascending=[True, False])

# Display results
print("\nResults (Sorted by ICR Percentile within Each Group):")
print(player_stats[['bat', 'group', 'ICR_percentile', 'ICR', 'match_count'] + [f'percentile_{metric}' for metric in metrics]])


################################## Final Implementation ##########################


import pandas as pd
import numpy as np
from scipy.stats import norm, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# ---------------- CONFIG ----------------
DEFAULT_PRIOR = {'rvap': 0.3, 'icr': 0.4, 'wpa': 0.3}
DEFAULT_ALPHA = 0.6
MIN_MATCHES_DEFAULT = 3
MIN_MATCHES_FALLBACK = 3
WINSOR_PCT = 0.01
MIN_GROUP_SIZE = 5

# ---------------- Helper Functions ----------------
def _winsorize_series(s, pct=WINSOR_PCT):
    # Convert to pandas Series if it's not already
    if not isinstance(s, pd.Series):
        s = pd.Series(s)

    # Check conditions more explicitly
    if pct <= 0 or len(s) < 5:
        return s

    # Check if all values are NaN
    if s.isna().all():
        return s

    # Calculate quantiles only on non-NaN values
    lo = s.quantile(pct)
    hi = s.quantile(1 - pct)

    # Apply winsorization
    return s.clip(lower=lo, upper=hi)

def _standardize_series(s):
    s = s.astype(float)
    mean = np.nanmean(s)
    std = np.nanstd(s, ddof=1)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mean) / std

def compute_weights_reliability(n1, n2, n3):
    s1 = np.sqrt(np.array(n1, dtype=float))
    s2 = np.sqrt(np.array(n2, dtype=float))
    s3 = np.sqrt(np.array(n3, dtype=float))
    denom = s1 + s2 + s3
    denom[denom == 0] = 1.0
    return s1/denom, s2/denom, s3/denom

def compute_weights_invvar(z1, z2, z3, n1_mean, n2_mean, n3_mean):
    v1 = np.nanvar(z1, ddof=1) if np.sum(~np.isnan(z1)) > 1 else 1.0
    v2 = np.nanvar(z2, ddof=1) if np.sum(~np.isnan(z2)) > 1 else 1.0
    v3 = np.nanvar(z3, ddof=1) if np.sum(~np.isnan(z3)) > 1 else 1.0
    w1 = n1_mean / v1
    w2 = n2_mean / v2
    w3 = n3_mean / v3
    arr = np.array([w1, w2, w3], dtype=float)
    s = arr.sum()
    return (arr / s).tolist() if s != 0 else [1/3, 1/3, 1/3]

def compute_weights_pca(z1, z2, z3):
    Z = np.vstack([z1, z2, z3]).T
    pca = PCA(n_components=1)
    pca.fit(np.nan_to_num(Z))
    load = np.abs(pca.components_[0])
    total = load.sum() if load.sum() != 0 else 1.0
    return (load / total).tolist(), pca.explained_variance_ratio_[0]

def compute_weights_regression(z1, z2, z3, y):
    df = pd.DataFrame({'z1': z1, 'z2': z2, 'z3': z3, 'y': y}).dropna()
    if df.shape[0] < 6:
        return [1/3, 1/3, 1/3], 0.0
    X = df[['z1', 'z2', 'z3']].values
    Xs = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=1) + 1e-9)
    ys = (df['y'].values - df['y'].mean()) / (df['y'].std(ddof=1) + 1e-9)
    reg = LinearRegression().fit(Xs, ys)
    betas = np.abs(reg.coef_)
    s = betas.sum()
    return (betas / s).tolist() if s != 0 else [1/3, 1/3, 1/3], reg.score(Xs, ys)

def prepare_merged_from_scores(
    results_df,
    player_stats,
    player_group_stats,
    id_results='player',
    id_player_stats='bat',
    id_group_stats='bat',
    rvap_col='rvap_score',
    icr_col='ICR',
    wpa_col='total_wpa',
    min_matches=MIN_MATCHES_DEFAULT
):
    r = results_df.copy()
    ps = player_stats.copy()
    pg = player_group_stats.copy()

    # Standardize player IDs
    r[id_results] = r[id_results].astype(str).str.strip().str.lower()
    ps[id_player_stats] = ps[id_player_stats].astype(str).str.strip().str.lower()
    pg[id_group_stats] = pg[id_group_stats].astype(str).str.strip().str.lower()

    # Rename ID columns to 'player'
    r = r.rename(columns={id_results: 'player'})
    ps = ps.rename(columns={id_player_stats: 'player'})
    pg = pg.rename(columns={id_group_stats: 'player'})

    # Validate required columns
    for df, col, name in [(pg, 'num_matches', 'player_group_stats'), (r, 'comp_group', 'results_df'),
                          (ps, 'group', 'player_stats'), (pg, 'group', 'player_group_stats')]:
        if col not in df.columns:
            raise KeyError(f"{name} must contain '{col}' column")

    # Align groups using primary_pos if available
    if 'primary_pos' in r.columns:
        def map_group(pos):
            try:
                pos = float(pos)
                if pos in [1, 2]:
                    return 'Openers'
                elif pos in [3, 4]:
                    return 'Top Order'
                elif pos in [5, 6]:
                    return 'Middle Order'
                elif pos in [7, 8]:
                    return 'Lower Order'
                return 'Other'
            except (ValueError, TypeError):
                return 'Other'
        r['comp_group'] = r['primary_pos'].apply(map_group)
        print("Aligned comp_group using primary_pos")

    # Check for common players
    r_players = set(r['player'].dropna().unique())
    ps_players = set(ps['player'].dropna().unique())
    pg_players = set(pg['player'].dropna().unique())
    common = r_players & ps_players & pg_players
    print(f"Unique players: results_df={len(r_players)}, player_stats={len(ps_players)}, player_group_stats={len(pg_players)}, common={len(common)}")
    if not common:
        raise ValueError("No common players across datasets. Check player IDs for consistency.")

    # Filter by min_matches
    pg_ok = pg[pg['num_matches'] >= min_matches].copy()
    if pg_ok.empty:
        print(f"Warning: No players with num_matches >= {min_matches}. Falling back to min_matches={MIN_MATCHES_FALLBACK}")
        pg_ok = pg[pg['num_matches'] >= MIN_MATCHES_FALLBACK].copy()
        if pg_ok.empty:
            raise ValueError(f"No players with num_matches >= {MIN_MATCHES_FALLBACK}. Check num_matches distribution.")
    eligible = common & set(pg_ok['player'].unique())
    print(f"Eligible players after min_matches filter: {len(eligible)}")
    if not eligible:
        raise ValueError(f"No players with num_matches >= {min_matches or MIN_MATCHES_FALLBACK}")

    # Filter DataFrames
    r = r[r['player'].isin(eligible)][['player', 'comp_group', rvap_col, 'total_innings']].copy()
    ps = ps[ps['player'].isin(eligible)][['player', 'group', icr_col, 'match_count']].copy()
    pg = pg[pg['player'].isin(eligible)][['player', 'group', wpa_col, 'num_matches']].copy()

    # Check group alignment
    group_mismatches = set(r['comp_group'].unique()) ^ set(ps['group'].unique())
    if group_mismatches:
        print(f"Warning: Group mismatches between comp_group and group: {group_mismatches}. Attempting merge with comp_group.")

    # Merge
    merged = r.merge(ps, left_on=['player', 'comp_group'], right_on=['player', 'group'], how='inner')
    merged = merged.merge(pg, left_on=['player', 'comp_group'], right_on=['player', 'group'], how='inner')
    merged = merged.drop(columns=['group_x', 'group_y'], errors='ignore')
    merged = merged.rename(columns={rvap_col: 'rvap_raw', icr_col: 'icr_raw', wpa_col: 'wpa_raw', 'total_innings': 'n1', 'match_count': 'n2', 'num_matches': 'n3'})
    print(f"Merged DataFrame size: {len(merged)}")

    if merged.empty:
        raise ValueError("Merged DataFrame is empty after filtering and merging. Check group alignment or data consistency.")

    for c in ['rvap_raw', 'icr_raw', 'wpa_raw', 'n1', 'n2', 'n3']:
        merged[c] = pd.to_numeric(merged[c], errors='coerce')

    merged = merged.dropna(subset=['rvap_raw', 'icr_raw', 'wpa_raw'], how='all').reset_index(drop=True)
    print(f"Final merged DataFrame size after dropping missing metrics: {len(merged)}")
    return merged

def combine_raw_scores_to_icr(
    results_df,
    player_stats,
    player_group_stats,
    id_results='player',
    id_player_stats='bat',
    id_group_stats='bat',
    rvap_col='rvap_score',
    icr_col='ICR',
    wpa_col='total_wpa',
    min_matches=MIN_MATCHES_DEFAULT,
    prior_weights=None,
    alpha=DEFAULT_ALPHA,
    method_primary='reliability',
    outcome_col='avg_wpa',
    winsor_pct=WINSOR_PCT
):
    if prior_weights is None:
        prior_weights = DEFAULT_PRIOR.copy()

    # Merge and restrict players
    merged = prepare_merged_from_scores(
        results_df, player_stats, player_group_stats,
        id_results, id_player_stats, id_group_stats,
        rvap_col, icr_col, wpa_col, min_matches
    )

    # Filter out 'Other' group
    merged = merged[merged['comp_group'] != 'Other'].copy()
    if merged.empty:
        raise ValueError("No players remain after filtering out 'Other' group. Check comp_group values.")

    # Dynamic winsorization
    merged[['rvap_w', 'icr_w', 'wpa_w']] = merged.groupby('comp_group')[['rvap_raw', 'icr_raw', 'wpa_raw']].transform(
        lambda x: _winsorize_series(x, pct=min(winsor_pct, 1/len(x)) if len(x) > 5 else 0))
    merged[['rvap_z', 'icr_z', 'wpa_z']] = merged.groupby('comp_group')[['rvap_w', 'icr_w', 'wpa_w']].transform(_standardize_series)

    # Handle counts
    merged[['n1', 'n2', 'n3']] = merged[['n1', 'n2', 'n3']].fillna(1.0).astype(float)

    # Compute weights by group
    weight_summary = {}
    merged['icr_z_final'] = 0.0
    merged['icr_percentile_final'] = np.nan
    print("Processing groups...")
    for group in merged['comp_group'].unique():
        mask = merged['comp_group'] == group
        group_data = merged[mask].copy() # Using a copy to avoid SettingWithCopyWarning
        group_size = len(group_data)
        print(f"Group {group}: {group_size} players")

        # New defensive check to handle cases where a group becomes empty after filtering
        if group_data.empty:
            print(f"Warning: Group {group} has no valid data after filtering. Skipping group.")
            continue

        if group_size < MIN_GROUP_SIZE:
            data_w_rel = data_w_inv = data_w_pca = data_w_reg = [1/3, 1/3, 1/3]
            pca_var = reg_r2 = 0.0
            print(f"Warning: Group {group} has {group_size} players < {MIN_GROUP_SIZE}. Using equal weights.")
        else:
            data_w_rel = compute_weights_reliability(group_data['n1'], group_data['n2'], group_data['n3'])
            data_w_rel = (np.nanmean(data_w_rel[0]), np.nanmean(data_w_rel[1]), np.nanmean(data_w_rel[2]))
            data_w_inv = compute_weights_invvar(group_data['rvap_z'], group_data['icr_z'], group_data['wpa_z'],
                                                np.nanmean(group_data['n1']), np.nanmean(group_data['n2']), np.nanmean(group_data['n3']))
            data_w_pca, pca_var = compute_weights_pca(group_data['rvap_z'], group_data['icr_z'], group_data['wpa_z'])

            y_data = group_data[outcome_col] if outcome_col in group_data.columns else group_data['wpa_raw']
            data_w_reg, reg_r2 = compute_weights_regression(group_data['rvap_z'], group_data['icr_z'], group_data['wpa_z'], y_data)


        data_w = {
            'reliability': data_w_rel,
            'invvar': data_w_inv,
            'pca': data_w_pca,
            'regression': data_w_reg
        }.get(method_primary, data_w_rel)

        final_w_raw = (
            alpha * prior_weights['rvap'] + (1 - alpha) * data_w[0],
            alpha * prior_weights['icr'] + (1 - alpha) * data_w[1],
            alpha * prior_weights['wpa'] + (1 - alpha) * data_w[2]
        )
        s = sum(final_w_raw)
        final_w = [x / s for x in final_w_raw] if s != 0 else [1/3, 1/3, 1/3]

        merged.loc[mask, 'icr_z_final'] = (final_w[0] * merged.loc[mask, 'rvap_z'] +
                                           final_w[1] * merged.loc[mask, 'icr_z'] +
                                           final_w[2] * merged.loc[mask, 'wpa_z'])
        merged.loc[mask, 'icr_percentile_final'] = norm.cdf(merged.loc[mask, 'icr_z_final']) * 100.0

        weight_summary[group] = {
            'prior': prior_weights,
            'alpha': alpha,
            'data_weights': {'reliability': data_w_rel, 'invvar': data_w_inv, 'pca': data_w_pca, 'regression': data_w_reg},
            'final_weights': {'rvap': final_w[0], 'icr': final_w[1], 'wpa': final_w[2]},
            'method_primary': method_primary,
            'pca_variance_explained': pca_var,
            'regression_r2': reg_r2,
            'group_size': group_size
        }
        print(f"Group {group}: Added to weight_summary with keys: {list(weight_summary[group].keys())}")

    # Compute percentiles
    merged['rvap_score_percentile'] = merged.groupby('comp_group')['rvap_raw'].rank(pct=True) * 100
    merged['icr_rank_final'] = merged.groupby('comp_group')['icr_percentile_final'].rank(method='min', ascending=False)

    # Validation
    corr_wpa = spearmanr(merged['icr_percentile_final'], merged['wpa_raw'], nan_policy='omit')[0]
    corr_rvap = spearmanr(merged['icr_percentile_final'], merged['rvap_raw'], nan_policy='omit')[0]
    weight_summary['validation'] = {
        'spearman_corr_with_wpa': corr_wpa,
        'spearman_corr_with_rvap': corr_rvap
    }
    print(f"Validation dictionary keys: {list(weight_summary['validation'].keys())}")

    # Sort by comp_group and icr_percentile_final
    merged = merged.sort_values(['comp_group', 'icr_percentile_final'], ascending=[True, False])

    # Print diagnostics
    print("\nWeight Summary:")
    for group in weight_summary:
        if group != 'validation':
            summary = weight_summary[group]
            print(f"\n{group} (Size: {summary['group_size']}):")
            print(f"  Prior Weights: {summary['prior']}")
            print(f"  Alpha: {summary['alpha']:.2f}")
            print(f"  Data-Driven Weights:")
            for method, weights in summary['data_weights'].items():
                print(f"    {method}: [rvap: {weights[0]:.3f}, icr: {weights[1]:.3f}, wpa: {weights[2]:.3f}]")
            print(f"  Final Weights: [rvap: {summary['final_weights']['rvap']:.3f}, icr: {summary['final_weights']['icr']:.3f}, wpa: {summary['final_weights']['wpa']:.3f}]")
            print(f"  PCA Variance Explained: {summary['pca_variance_explained']:.3f}")
            print(f"  Regression R²: {summary['regression_r2']:.3f}")
    print(f"\nValidation: Spearman Corr with wpa_raw: {corr_wpa:.3f}, rvap_raw: {corr_rvap:.3f}")
    print("\nMetric Correlations:")
    print(merged[['rvap_raw', 'icr_raw', 'wpa_raw']].corr(method='spearman'))

    return merged, weight_summary

# ---------------- USAGE EXAMPLE ----------------
try:
    merged, summary = combine_raw_scores_to_icr(
        results_df, player_stats, player_group_stats,
        id_results='player', id_player_stats='bat', id_group_stats='bat',
        rvap_col='rvap_score', icr_col='ICR', wpa_col='total_wpa',
        min_matches=5, prior_weights={'rvap': 0.2, 'icr': 0.5, 'wpa': 0.3},
        alpha=0.6, method_primary='reliability', outcome_col='avg_wpa', winsor_pct=0.01
    )
    print("\nResults (Sorted by icr_percentile_final within Each comp_group):")
    print(merged[['player', 'comp_group', 'icr_percentile_final', 'rvap_score_percentile', 'icr_rank_final', 'rvap_raw', 'icr_raw', 'wpa_raw', 'n1', 'n2', 'n3']])
except ValueError as e:
    print(f"ValueError: {e}")
except KeyError as e:
    print(f"KeyError: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")


import pandas as pd
import numpy as np

# parameter for shrinkage strength
K = 5.0

df = merged.copy()

# 1) rename originals
if 'icr_z_final' in df.columns:
    df = df.rename(columns={'icr_z_final': 'icr_z_final_initial'})
else:
    df['icr_z_final_initial'] = np.nan

if 'icr_percentile_final' in df.columns:
    df = df.rename(columns={'icr_percentile_final': 'icr_percentile_final_initial'})
else:
    df['icr_percentile_final_initial'] = np.nan

# ensure numeric
df['icr_z_final_initial'] = pd.to_numeric(df['icr_z_final_initial'], errors='coerce')
df['n1'] = pd.to_numeric(df['n1'], errors='coerce').fillna(0).astype(float)

# 2) global mean (since no group info here)
global_mu = df['icr_z_final_initial'].mean(skipna=True)

# 3) shrinkage
def shrink_icr(row, k=K, mu=global_mu):
    x = row['icr_z_final_initial']
    n = row['n1']
    if pd.isna(x):
        return np.nan
    if (n is None) or (n <= 0) or pd.isna(n):
        alpha = 0.0
    else:
        alpha = float(n) / (float(n) + float(k))
    return alpha * x + (1.0 - alpha) * mu

df['icr_z_final'] = df.apply(shrink_icr, axis=1)

# 4) recompute percentiles (global, not group-based)
def compute_percentile(s):
    non_na = s.dropna()
    n = len(non_na)
    if n == 0:
        return pd.Series([np.nan] * len(s), index=s.index)
    ranks = s.rank(method='average', na_option='keep')
    if n == 1:
        pct = ranks.where(ranks.isna(), 100.0)
    else:
        pct = (ranks - 1.0) / (n - 1.0) * 100.0
    pct = pct.where(~s.isna(), np.nan)
    return pct

df['icr_percentile_final'] = compute_percentile(df['icr_z_final'])

# 5) diagnostics
print(f"Shrinkage applied with K={K}. Sample:")
print(df[['player', 'n1', 'icr_z_final_initial', 'icr_z_final',
          'icr_percentile_final_initial', 'icr_percentile_final']].head(10).to_string(index=False))

# final output
merged = df
df=DF

merged=merged.sort_values('icr_percentile_final', ascending = False)

icr_df=merged

icr_df['batter'] = icr_df['player'].astype(str).str.strip().str.title().replace({'nan': np.nan})



# Ensure lowercase for consistent merge
icr_df['player_lower'] = icr_df['player'].str.lower()
df['bat_lower'] = df['bat'].str.lower()

# Merge on lowercase version
icr_df = icr_df.merge(df[['bat_lower', 'team_bat']],
                      left_on='player_lower', right_on='bat_lower',
                      how='left')

# Drop helper column if not needed
icr_df = icr_df.drop(columns=['player_lower', 'bat_lower'])


icr_df=icr_df.drop_duplicates(subset=['player'])

icr_df[['batter','team_bat','icr_percentile_final']].head(10)

icr_25 = icr_df




########## BOWLING ICR CALCULATION #####################


import numpy as np
import pandas as pd

# ----------------------------
# Configure & defensive copy
# ----------------------------
# df is expected to be your ball-by-ball DataFrame already loaded
df = df.copy()

# Use floor on 'over' (True) or use raw numeric comparison (False)
use_floor = True

# Ensure the critical columns exist
required_cols = ['p_match', 'inns', 'ball_id', 'over']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise KeyError(f"Missing required columns in df: {missing}")

# Make sure over is numeric (coerce bad values to NaN)
df['over'] = pd.to_numeric(df['over'], errors='coerce')

# ----------------------------
# 1) Sort the dataframe
# ----------------------------
df = df.sort_values(['p_match', 'inns', 'ball_id']).reset_index(drop=True)

# ----------------------------
# 2) Build over_index (integer over number)
#    If 'over' contains floats like 6.1 (ball notation),
#    using floor( ) maps 6.1 -> 6 (recommended).
# ----------------------------
if use_floor:
    # floor, then cast to integer (use Int64 to allow NaN)
    df['over_idx'] = np.floor(df['over']).astype('Int64')
else:
    # use rounded integer part or raw value depending on dataset semantics
    # here we take the integer part (trunc) if you want 6.1 -> 6. Change if required.
    df['over_idx'] = df['over'].astype('Int64')

# ----------------------------
# 3) Map to phase
# ----------------------------
def map_phase(over_idx):
    # Handles pandas NA (pd.NA) and numpy nan
    if pd.isna(over_idx):
        return 'unknown'
    o = int(over_idx)
    if o <= 6:
        return 'Powerplay'
    elif o <= 15:
        return 'Middle'
    else:
        return 'Death'

df['phase'] = df['over_idx'].apply(map_phase)

# ----------------------------
# 4) Quick sanity checks / outputs
# ----------------------------
print("Phase value counts:")
print(df['phase'].value_counts(dropna=False))

# Show a few sample rows to verify
display_cols = ['p_match', 'inns', 'ball_id', 'over', 'over_idx', 'phase']
print("\nSample rows (first 30):")
print(df[display_cols].head(30).to_string(index=False))


#!/usr/bin/env python3
"""
Bowling RVAP (B-RVAP) — multi-group, econ-based deltas, wicket->run median-from-dismissals,
plus Runs_Won_Back metric (added).

- Input: ball-by-ball DataFrame `df` with required columns:
  ['p_match','inns','bowl','venue','date','score','out']
  Optional: ['noball','wide','bowl_kind','phase','venue_factor','bat'].
- Only bowlers with >= min_balls_threshold legal balls are included (default 70).
- Peer groups: (venue, bowl_kind, phase_group) where bowler is in phase_group if his % in that phase >= median% (eligible bowlers).
- Runs_Won_Back: per-innings metric built from batters who faced the bowler that innings.
"""

import numpy as np
import pandas as pd
from datetime import datetime

PHASE_LIST = ['Powerplay', 'Middle', 'Death']
MIN_BALLS_THRESHOLD = 70

def log_progress(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def calculate_bowler_rvap(
    df,
    player_name=None,
    VOLUME_WEIGHT=1.0,
    WICKET_EQUIVALENT=None,   # if None => compute median runs per dismissed batter-innings from df
    min_peer_innings=3,
    min_balls_threshold=MIN_BALLS_THRESHOLD,
    detailed=False
):
    log_progress("Starting Bowling RVAP calculation (multi-group, econ deltas, Runs_Won_Back)")

    df = df.copy()

    # -------------------------
    # Validate required columns
    # -------------------------
    req = ['p_match','inns','bowl','venue','date','score','out']
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Ensure bowl_kind present
    if 'bowl_kind' not in df.columns:
        if 'bowl_style' in df.columns:
            df['bowl_kind'] = df['bowl_style'].apply(
                lambda s: 'spin' if isinstance(s, str) and 'spin' in s.lower() else 'pace')
            log_progress("Inferred 'bowl_kind' from 'bowl_style'")
        else:
            df['bowl_kind'] = 'pace'
            log_progress("No 'bowl_kind' or 'bowl_style' found — defaulting bowl_kind to 'pace'")

    # Ensure phase present
    if 'phase' not in df.columns:
        df['phase'] = 'Unknown'

    # extras & legal
    for c in ['noball','wide']:
        if c not in df.columns:
            df[c] = 0
    df['noball'] = pd.to_numeric(df['noball'], errors='coerce').fillna(0).astype(int)
    df['wide'] = pd.to_numeric(df['wide'], errors='coerce').fillna(0).astype(int)
    df['legal'] = (~((df['noball'].astype(bool)) | (df['wide'].astype(bool)))).astype(int)

    # normalize bowl names
    df['bowl'] = df['bowl'].astype(str).str.strip()
    df = df[df['bowl'] != '']

    # -------------------------
    # Compute wicket-equivalent
    #    Use median runs per **batter-innings that ended with a dismissal** (out_sum > 0)
    # -------------------------
    if WICKET_EQUIVALENT is None:
        wicket_value = None
        if 'bat' in df.columns:
            bat_inns = (
                df.groupby(['p_match','inns','bat'], as_index=False)
                .agg(runs_innings=('score','sum'), balls_innings=('legal','sum'), out_sum=('out','sum'))
            )
            bat_inns_valid = bat_inns[(bat_inns['balls_innings'] > 0) & (bat_inns['out_sum'] > 0)]
            if len(bat_inns_valid) > 0:
                wicket_value = float(bat_inns_valid['runs_innings'].median())
                log_progress(f"Computed wicket-equivalent = median runs of dismissed batter-innings = {wicket_value:.3f}")
            else:
                # fallback to any batter-innings with balls > 0
                bat_inns_any = bat_inns[bat_inns['balls_innings'] > 0]
                if len(bat_inns_any) > 0:
                    wicket_value = float(bat_inns_any['runs_innings'].median())
                    log_progress(f"No dismissed-batter innings found; fallback median across all batter-innings = {wicket_value:.3f}")
        if wicket_value is None:
            # fallback to team-innings median
            team_inns = (
                df.groupby(['p_match','inns'], as_index=False)
                .agg(runs_innings=('score','sum'), balls_innings=('legal','sum'))
            )
            team_valid = team_inns[team_inns['balls_innings'] > 0]
            if len(team_valid) > 0:
                wicket_value = float(team_valid['runs_innings'].median())
                log_progress(f"No batter info; using team-innings median runs = {wicket_value:.3f}")
            else:
                wicket_value = 20.0
                log_progress("No innings found to compute median; using fallback 20.0 runs/wicket")
    else:
        wicket_value = float(WICKET_EQUIVALENT)
        log_progress(f"Using user-supplied WICKET_EQUIVALENT = {wicket_value:.3f}")

    # -------------------------
    # Precompute bowler total legal balls; apply min_balls_threshold filter for eligible bowlers
    # -------------------------
    bowler_ball_totals = (
        df[df['legal'] == 1]
        .groupby('bowl', as_index=False)
        .agg(total_legal_balls=('legal','sum'))
    )
    eligible_bowlers = set(bowler_ball_totals.loc[bowler_ball_totals['total_legal_balls'] >= min_balls_threshold, 'bowl'].tolist())
    log_progress(f"Found {len(eligible_bowlers)} eligible bowlers with >= {min_balls_threshold} legal balls")

    if len(eligible_bowlers) == 0:
        log_progress("No eligible bowlers found — returning empty DataFrame")
        return pd.DataFrame(columns=[
            'bowler','total_runs_conceded','total_innings','actual_innings',
            'total_legal_balls','runs_won_back','volume_adjustment','total_venue_wicket_deltas','venues_played',
            'b_rvap_score','bowl_kind','phase_groups'
        ])

    # -------------------------
    # Prepare batter average map for Runs_Won_Back (if 'bat' exists)
    # -------------------------
    if 'bat' in df.columns:
        batter_inns = (
            df.groupby(['p_match','inns','bat'], as_index=False)
            .agg(runs_innings=('score','sum'), balls_innings=('legal','sum'))
        )
        batter_inns_valid = batter_inns[batter_inns['balls_innings'] > 0]
        if len(batter_inns_valid) > 0:
            batter_avg = batter_inns_valid.groupby('bat', as_index=False)['runs_innings'].mean().rename(columns={'runs_innings':'batter_avg_runs'})
            # global median fallback
            global_median_batter = float(batter_inns_valid['runs_innings'].median())
        else:
            batter_avg = pd.DataFrame(columns=['bat','batter_avg_runs'])
            global_median_batter = float(20.0)
    else:
        batter_avg = pd.DataFrame(columns=['bat','batter_avg_runs'])
        global_median_batter = float(20.0)

    batter_avg_map = dict(zip(batter_avg['bat'], batter_avg['batter_avg_runs']))

    # -------------------------
    # 1) Phase % per bowler -> assign to phase groups (>= median%)
    #    compute using only eligible bowlers
    # -------------------------
    phase_counts = (
        df[(df['legal'] == 1) & (df['bowl'].isin(eligible_bowlers))]
        .groupby(['bowl','phase'], as_index=False)
        .agg(balls_phase=('legal','sum'))
    )
    total_balls_bowler = phase_counts.groupby('bowl', as_index=False).agg(total_balls=('balls_phase','sum'))
    phase_counts = phase_counts.merge(total_balls_bowler, on='bowl', how='left')
    phase_counts['phase_frac'] = phase_counts['balls_phase'] / phase_counts['total_balls'].replace(0, np.nan)

    phase_pct = phase_counts.pivot_table(index='bowl', columns='phase', values='phase_frac', fill_value=0).reset_index()
    for p in PHASE_LIST:
        if p not in phase_pct.columns:
            phase_pct[p] = 0.0

    median_phase = phase_pct[PHASE_LIST].median()
    log_progress(f"Median phase fractions (eligible bowlers): {median_phase.to_dict()}")

    group_rows = []
    for _, row in phase_pct.iterrows():
        bowler = row['bowl']
        for phase in PHASE_LIST:
            if row[phase] >= median_phase[phase]:
                group_rows.append({'bowl': bowler, 'phase_group': phase})
    bowler_groups = pd.DataFrame(group_rows)
    if bowler_groups.empty:
        log_progress("No bowlers meet the >= median% condition for any phase (empty groups)")

    # Add bowl_kind
    bowler_kind = (
        df[df['bowl'].isin(eligible_bowlers)]
        .groupby('bowl', as_index=False)['bowl_kind']
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'pace')
    )
    bowler_groups = bowler_groups.merge(bowler_kind, on='bowl', how='left')

    bowler_to_groups = bowler_groups.groupby('bowl')['phase_group'].apply(list).to_dict()

    # -------------------------
    # 2) Build per-innings aggregation (original, one row per bowler-inning) for eligible bowlers
    # -------------------------
    agg_innings = df[df['bowl'].isin(eligible_bowlers)].groupby(
        ['p_match','inns','bowl','venue','bowl_kind'], as_index=False
    ).agg(
        runs_conceded=('cur_bowl_runs','max'),   # use final cumulative runs conceded
        balls=('legal','sum'),
        wickets=('cur_bowl_wkts','max'),         # ✅ final wickets from cumulative column
        venue_factor=('venue_factor','first'),
        date=('date','first')
    )


    if 'venue_factor' in agg_innings.columns:
        agg_innings['venue_factor'] = pd.to_numeric(agg_innings['venue_factor'], errors='coerce').replace(0, np.nan).fillna(1.0)
    else:
        agg_innings['venue_factor'] = 1.0

    # -------------------------
    # 3) Build peer stats using expanded per-innings table (bowler_innings x phase_group)
    #    and compute peer expected economy (runs per over)
    # -------------------------
    if not bowler_groups.empty:
        expanded_for_peers = agg_innings.merge(bowler_groups, on=['bowl','bowl_kind'], how='left')
    else:
        expanded_for_peers = agg_innings.copy()
        expanded_for_peers['phase_group'] = np.nan

    peer_stats = (
        expanded_for_peers.groupby(['venue','bowl_kind','phase_group'], as_index=False)
        .agg(
            total_balls=('balls','sum'),
            total_wickets=('wickets','sum'),
            n_innings=('balls','count'),
            sum_runs=('runs_conceded','sum')
        )
    )
    peer_stats['group_bpd'] = peer_stats.apply(
        lambda r: (r['total_balls'] / r['total_wickets']) if r['total_wickets'] > 0 else np.nan, axis=1)
    peer_stats['peer_avg_adj_runs_per_innings'] = peer_stats['sum_runs'] / peer_stats['n_innings'].replace(0, np.nan)
    peer_stats['peer_expected_econ'] = peer_stats.apply(
        lambda r: (r['sum_runs'] * 6.0 / r['total_balls']) if (r['total_balls'] > 0) else np.nan, axis=1)

    peer_lookup = {}
    for _, row in peer_stats.iterrows():
        key = (row['venue'], row['bowl_kind'], row['phase_group'])
        peer_lookup[key] = {
            'group_bpd': row['group_bpd'],
            'peer_avg_adj_runs_per_innings': row['peer_avg_adj_runs_per_innings'],
            'peer_expected_econ': row['peer_expected_econ'],
            'peer_n_innings': int(row['n_innings'])
        }

    # -------------------------
    # 4) Compute expected_wickets per original innings & economy metrics & Runs_Won_Back
    # -------------------------
    def expected_wickets_row(r):
        b = r['bowl']; venue = r['venue']; bkind = r['bowl_kind']
        groups = bowler_to_groups.get(b, [])
        exp_list = []
        for pg in groups:
            stats = peer_lookup.get((venue, bkind, pg), None)
            if stats is None:
                continue
            g_bpd = stats.get('group_bpd', np.nan)
            if pd.isna(g_bpd) or g_bpd == 0:
                continue
            exp_list.append(r['balls'] / g_bpd)
        if len(exp_list) == 0:
            return 0.0
        return float(np.nanmean(exp_list))

    agg_innings['expected_wickets'] = agg_innings.apply(expected_wickets_row, axis=1)
    agg_innings['wicket_delta'] = agg_innings['wickets'] - agg_innings['expected_wickets']

    # player economy (runs per over)
    def player_econ_calc(row):
        if row['balls'] > 0:
            return (row['runs_conceded'] * 6.0) / row['balls']
        return np.nan
    agg_innings['player_econ'] = agg_innings.apply(player_econ_calc, axis=1)

    # expected econ for that innings = mean of peer_expected_econ across bowler's groups at that venue/bowl_kind
    def expected_econ_row(r):
        b = r['bowl']; venue = r['venue']; bkind = r['bowl_kind']
        groups = bowler_to_groups.get(b, [])
        vals = []
        for pg in groups:
            stats = peer_lookup.get((venue, bkind, pg), None)
            if stats is None:
                continue
            val = stats.get('peer_expected_econ', np.nan)
            if not pd.isna(val):
                vals.append(val)
        if len(vals) == 0:
            return np.nan
        return float(np.nanmean(vals))

    agg_innings['expected_econ'] = agg_innings.apply(expected_econ_row, axis=1)
    agg_innings['econ_delta'] = agg_innings.apply(
        lambda r: (r['expected_econ'] - r['player_econ']) if (not pd.isna(r['expected_econ']) and not pd.isna(r['player_econ'])) else np.nan,
        axis=1
    )

    agg_innings['adj_runs'] = agg_innings['runs_conceded'] / agg_innings['venue_factor'].replace(np.nan, 1.0)

    # -------------------------
    # Runs_Won_Back: compute per original innings (for eligible bowlers only)
    # -------------------------
    # Precompute df_eligible subset for speedy lookups
    df_eligible = df[df['bowl'].isin(eligible_bowlers)].copy()

    def runs_won_back_for_row(r):
        # r: a row from agg_innings
        p_match = r['p_match']; inns = r['inns']; bowler = r['bowl']
        # rows where this bowler bowled in this innings
        sub = df_eligible[(df_eligible['p_match']==p_match) & (df_eligible['inns']==inns) & (df_eligible['bowl']==bowler)]
        if sub.shape[0] == 0:
            return 0.0
        # runs faced off this bowler by batter in this innings
        per_bat = sub.groupby('bat', as_index=False).agg(runs_off_bowler=('score','sum'), balls_off_bowler=('legal','sum'), out_sum=('out','sum'))
        # dismissed_by_bowler if out_sum > 0 (the out flag on that delivery)
        runs_wb_sum_dismissed = 0.0
        runs_sum_not_out = 0.0
        for _, br in per_bat.iterrows():
            batter = br['bat']
            runs_off = float(br['runs_off_bowler'])
            out_sum = int(br['out_sum'])
            # batter average runs per innings from dataset (fallback to global median if unknown)
            batter_avg_val = batter_avg_map.get(batter, global_median_batter)
            if out_sum > 0:
                # dismissed by this bowler
                runs_wb_sum_dismissed += (batter_avg_val - runs_off)
            else:
                # not dismissed by this bowler (but faced him) => subtract runs they made off him
                runs_sum_not_out += runs_off
        runs_wb = runs_wb_sum_dismissed - runs_sum_not_out
        # numeric safety
        return float(runs_wb)

    agg_innings['runs_won_back'] = agg_innings.apply(runs_won_back_for_row, axis=1)

    # -------------------------
    # 5) Per-bowler aggregation — totals computed from ORIGINAL data (not duplicated)
    #    - total_runs_conceded = sum(df['score']) for eligible bowlers
    #    - total_innings = nunique(df['p_match']) for eligible bowlers
    # -------------------------
    total_runs_by_bowler = df_eligible.groupby('bowl', as_index=False)['score'].sum().rename(columns={'score':'total_runs_conceded'})
    total_innings_by_bowler = df_eligible.groupby('bowl', as_index=False)['p_match'].nunique().rename(columns={'p_match':'total_innings'})
    actual_innings_count = agg_innings.groupby('bowl', as_index=False).agg(actual_innings=('p_match','count'))

    result_rows = []
    for b in sorted(agg_innings['bowl'].unique()):
        br = agg_innings[agg_innings['bowl'] == b].copy()

        total_runs = float(total_runs_by_bowler.loc[total_runs_by_bowler['bowl']==b, 'total_runs_conceded'].iloc[0]) \
                     if (b in total_runs_by_bowler['bowl'].values) else br['runs_conceded'].sum()
        total_innings_val = int(total_innings_by_bowler.loc[total_innings_by_bowler['bowl']==b, 'total_innings'].iloc[0]) \
                            if (b in total_innings_by_bowler['bowl'].values) else br.shape[0]
        total_wicket_deltas = float(br['wicket_delta'].sum())
        venues_played = int(br['venue'].nunique())

        # Runs_Won_Back aggregated across innings
        runs_won_back_total = float(br['runs_won_back'].sum())

        # Volume adjustment: compute using economy deltas per venue
        venue_summary = br.groupby('venue').agg(
            player_avg_econ = ('player_econ', lambda x: float(np.nanmean(x)) if len(x)>0 else np.nan),
            n_innings = ('p_match','nunique'),
            bowl_kind = ('bowl_kind','first')
        ).reset_index()

        def compute_peer_econ_for_row(r):
            pg_list = bowler_to_groups.get(b, [])
            vals = []
            for pg in pg_list:
                val = peer_lookup.get((r['venue'], r['bowl_kind'], pg), {}).get('peer_expected_econ', np.nan)
                if not pd.isna(val):
                    vals.append(val)
            if len(vals) == 0:
                return np.nan
            return float(np.nanmean(vals))

        venue_summary['peer_avg_econ'] = venue_summary.apply(compute_peer_econ_for_row, axis=1)
        venue_summary['venue_term'] = venue_summary.apply(
            lambda r: (r['peer_avg_econ'] - r['player_avg_econ']) if (not pd.isna(r['peer_avg_econ']) and not pd.isna(r['player_avg_econ'])) else 0.0,
            axis=1
        )

        volume_adjustment = float(venue_summary['venue_term'].sum())

        # NEW: include runs_won_back_total in wicket-run term (as requested)
        wicket_term_runs = (wicket_value * total_wicket_deltas) + runs_won_back_total

        # ==== CONSISTENCY BONUS ADDED HERE ====
        total_legal_balls = int(bowler_ball_totals.loc[bowler_ball_totals['bowl']==b,'total_legal_balls'].iloc[0]) if b in bowler_ball_totals['bowl'].values else 0
        consistency_bonus = np.log1p(total_legal_balls) / np.log1p(min_balls_threshold)
        # ======================================

        b_rvap_score = wicket_term_runs + (VOLUME_WEIGHT * volume_adjustment) + 195 *consistency_bonus

        primary_kind = br['bowl_kind'].mode().iloc[0] if not br['bowl_kind'].mode().empty else None
        phase_groups_b = bowler_to_groups.get(b, [])


        result_rows.append({
            'bowler': b,
            'total_runs_conceded': total_runs,
            'total_innings': total_innings_val,
            'actual_innings': int(br.shape[0]),
            'total_legal_balls': int(bowler_ball_totals.loc[bowler_ball_totals['bowl']==b,'total_legal_balls'].iloc[0]) if b in bowler_ball_totals['bowl'].values else 0,
            'runs_won_back': runs_won_back_total,
            'volume_adjustment': volume_adjustment,
            'total_venue_wicket_deltas': total_wicket_deltas,
            'venues_played': venues_played,
            'consistency_bonus': consistency_bonus,
            'b_rvap_score': b_rvap_score,
            'bowl_kind': primary_kind,
            'phase_groups': phase_groups_b
        })

    result_df = pd.DataFrame(result_rows).sort_values('b_rvap_score', ascending=False).reset_index(drop=True)

    # -------------------------
    # Detailed section (optional)
    # -------------------------
    if detailed and player_name:
        match = None
        for b in result_df['bowler']:
            if b.lower() == player_name.lower() or player_name.lower() in b.lower():
                match = b
                break
        if match is None:
            print(f"[WARN] Bowler '{player_name}' not found among eligible bowlers.")
        else:
            br = agg_innings[agg_innings['bowl']==match].sort_values(['date','p_match','inns']).reset_index(drop=True)
            print("\n" + "="*100)
            print(f"DETAILED B-RVAP FOR: {match}")
            print("="*100)
            rr = result_df[result_df['bowler']==match].iloc[0]
            print(f"Summary: total_runs={rr['total_runs_conceded']:.1f}, total_innings (nunique p_match)={rr['total_innings']}, actual_innings_rows={rr['actual_innings']}, total_wicket_delta={rr['total_venue_wicket_deltas']:.3f}, total_legal_balls={rr['total_legal_balls']}")
            print(f"Using wicket-equivalent (median runs per dismissed batter-innings) = {wicket_value:.3f}")
            print(f"Phase groups: {rr['phase_groups']}")
            print(f"Runs_Won_Back (sum across innings) = {rr['runs_won_back']:.3f}")
            print("-"*100)
            display_cols = ['p_match','inns','date','venue','balls','wickets','runs_conceded','venue_factor','expected_wickets','wicket_delta','player_econ','expected_econ','econ_delta','runs_won_back','adj_runs']
            br_display = br[display_cols].copy()
            br_display[['expected_wickets','wicket_delta','player_econ','expected_econ','econ_delta','runs_won_back','adj_runs']] = br_display[['expected_wickets','wicket_delta','player_econ','expected_econ','econ_delta','runs_won_back','adj_runs']].round(3)
            print("Per-innings breakdown (original innings rows):")
            print(br_display.to_string(index=False))
            print("-"*100)
            vs = br.groupby('venue').agg(
                n_innings=('p_match','nunique'),
                sum_runs=('runs_conceded','sum'),
                sum_adj_runs=('adj_runs','sum'),
                sum_wickets=('wickets','sum'),
                sum_expected=('expected_wickets','sum'),
                sum_delta=('wicket_delta','sum'),
                avg_player_econ=('player_econ', lambda x: float(np.nanmean(x)) if len(x)>0 else np.nan),
                avg_expected_econ=('expected_econ', lambda x: float(np.nanmean(x)) if len(x)>0 else np.nan),
                sum_runs_won_back=('runs_won_back', lambda x: float(np.nansum(x)) if len(x)>0 else 0.0)
            ).reset_index()
            vs['peer_avg_adj_runs_per_innings'] = vs.apply(
                lambda r: np.nanmean([peer_lookup.get((r['venue'], rr['bowl_kind'], pg), {}).get('peer_avg_adj_runs_per_innings', np.nan)
                                      for pg in bowler_to_groups.get(match, [])]) if len(bowler_to_groups.get(match, []))>0 else np.nan,
                axis=1
            )
            print("Per-venue summary (player vs peer averages; econ in runs-per-over):")
            print(vs.to_string(index=False))
            print("-"*100)
            print("FINAL:")
            print(f"  total_venue_wicket_deltas = {rr['total_venue_wicket_deltas']:.3f}")
            print(f"  runs_won_back_total = {rr['runs_won_back']:.3f}")
            print(f"  volume_adjustment (sum of econ peer - player) = {rr['volume_adjustment']:.3f}")
            print(f"  wicket_equivalent (median runs/inning from dismissed batters) = {wicket_value:.3f}, VOLUME_WEIGHT = {VOLUME_WEIGHT}")
            print(f"  b_rvap_score = {rr['b_rvap_score']:.3f}")
            print("="*100)

    log_progress("Completed Bowling RVAP computation (multi-group, econ deltas, Runs_Won_Back)")
    return result_df

# -----------------------
# Example usage (run in your notebook)
# -----------------------
if __name__ == "__main__":
    # show Jasprit Bumrah if present and eligible
    try:
        res = calculate_bowler_rvap(df, player_name='Jasprit Bumrah', VOLUME_WEIGHT=0.5, detailed=True)
        print("\nTop 10 bowlers by b_rvap_score (eligible bowlers):")
        print(res.head(10).to_string(index=False))
    except Exception as e:
        print("Error running calculate_bowler_rvap:", str(e))


results_df = res

df['legal'] = (~((df['noball'].astype(bool)) | (df['wide'].astype(bool)))).astype(int)

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# -----------------------
# CONFIG
# -----------------------
PHASE_LIST = ['Powerplay', 'Middle', 'Death']
MIN_BALLS = 70
EPS = 1e-9

# -----------------------
# Defensive copy
# -----------------------
df = df.copy()

# Ensure legal flag exists
if 'legal' not in df.columns:
    for c in ['noball','wide']:
        if c not in df.columns:
            df[c] = 0
    df['legal'] = (~((df['noball'].astype(bool)) | (df['wide'].astype(bool)))).astype(int)
else:
    df['legal'] = pd.to_numeric(df['legal'], errors='coerce').fillna(0).astype(int)

# Normalize bowler name
df['bowl'] = df['bowl'].astype(str).str.strip()

# Build runs attributed to bowler per-ball (use 'bowlruns' if present; else fallback)
if 'bowlruns' in df.columns:
    df['runs_bowler'] = pd.to_numeric(df['bowlruns'], errors='coerce').fillna(0.0)
else:
    # exclude byes/legbyes from bowler conceded runs
    df['byes'] = pd.to_numeric(df.get('byes', 0), errors='coerce').fillna(0)
    df['legbyes'] = pd.to_numeric(df.get('legbyes', 0), errors='coerce').fillna(0)
    df['score'] = pd.to_numeric(df.get('score', 0), errors='coerce').fillna(0)
    df['runs_bowler'] = df['score'] - df['byes'] - df['legbyes']
    df['runs_bowler'] = df['runs_bowler'].fillna(0.0)

# -----------------------
# Eligible bowlers: >= MIN_BALLS legal deliveries
# -----------------------
balls_by_bowler = df[df['legal'] == 1].groupby('bowl', as_index=False)['legal'].sum().rename(columns={'legal':'total_legal_balls'})
eligible_bowlers = set(balls_by_bowler.loc[balls_by_bowler['total_legal_balls'] >= MIN_BALLS, 'bowl'].tolist())
print(f"Eligible bowlers (>= {MIN_BALLS} legal balls): {len(eligible_bowlers)}")

if len(eligible_bowlers) == 0:
    raise ValueError("No eligible bowlers found (increase data or lower MIN_BALLS).")

# Work with eligible subset for phase medians etc.
df_elig = df[df['bowl'].isin(eligible_bowlers)].copy()

# -----------------------
# Phase % per bowler (eligible bowlers only)
# -----------------------
phase_counts = (
    df_elig[df_elig['legal'] == 1]
    .groupby(['bowl','phase'], as_index=False)['legal']
    .sum()
    .rename(columns={'legal':'balls_in_phase'})
)

total_balls = phase_counts.groupby('bowl', as_index=False)['balls_in_phase'].sum().rename(columns={'balls_in_phase':'total_balls'})
phase_counts = phase_counts.merge(total_balls, on='bowl', how='left')
phase_counts['phase_pct'] = 100.0 * phase_counts['balls_in_phase'] / (phase_counts['total_balls'] + EPS)

# Pivot to wide
phase_wide = phase_counts.pivot_table(index='bowl', columns='phase', values='phase_pct', fill_value=0.0).reset_index()
# Ensure canonical phase columns
for p in PHASE_LIST:
    if p not in phase_wide.columns:
        phase_wide[p] = 0.0

# Summary statistics across eligible bowlers
phase_summary = pd.DataFrame(index=PHASE_LIST)
phase_summary['Min %'] = [phase_wide[p].min() for p in PHASE_LIST]
phase_summary['Max %'] = [phase_wide[p].max() for p in PHASE_LIST]
phase_summary['Avg %'] = [phase_wide[p].mean() for p in PHASE_LIST]
phase_summary['Median %'] = [phase_wide[p].median() for p in PHASE_LIST]
print("\nSummary statistics of phase % across eligible bowlers:")
print(phase_summary.round(2))

# Build median dict for grouping
phase_medians = phase_summary['Median %'].to_dict()

# For ease, join phase_wide into an agg base
# We'll compute per-bowler aggregates from df_elig (per-ball sums) and then attach phase columns
agg = (
    df_elig.groupby('bowl', as_index=False)
    .agg(
        runs_conceded=('runs_bowler','sum'),
        balls=('legal','sum'),
        wickets=('out','sum'),
        match_count=('p_match','nunique')
    )
)

# Only keep bowlers meeting MIN_BALLS (defensive, though eligible_bowlers already filtered)
agg = agg[agg['balls'] >= MIN_BALLS].reset_index(drop=True)

# Merge phase % columns
agg = agg.merge(phase_wide[['bowl'] + PHASE_LIST], on='bowl', how='left').fillna(0.0)

# -----------------------
# Improved grouping logic (phase_groups + primary_group)
# -----------------------
def get_groups_and_primary(row):
    groups = []
    excess = {}
    for p in PHASE_LIST:
        my_pct = float(row.get(p, 0.0))
        med = float(phase_medians.get(p, 0.0))
        if my_pct >= med - 1e-12:   # >= median (numerical safety)
            groups.append(p)
            excess[p] = my_pct - med
    if len(groups) == 0:
        return pd.Series({'phase_groups': ['Mixed'], 'primary_group': 'Mixed'})
    # pick phase with max positive excess as primary
    primary = max(excess.items(), key=lambda kv: kv[1])[0]
    return pd.Series({'phase_groups': groups, 'primary_group': primary})

gp = agg.apply(get_groups_and_primary, axis=1)
agg['phase_groups'] = gp['phase_groups']
agg['primary_group'] = gp['primary_group']

# -----------------------
# Compute derived bowling metrics
# -----------------------
agg['overs'] = agg['balls'] / 6.0
agg['Econ'] = agg['runs_conceded'] / (agg['overs'] + EPS)               # lower better
agg['SR'] = agg['balls'] / (agg['wickets'] + EPS)                      # lower better (balls per wicket)
agg['Avg'] = agg['runs_conceded'] / (agg['wickets'] + EPS)             # lower better
agg['Wkts_per_6'] = agg['wickets'] / (agg['balls']/6.0 + EPS)          # higher better

# Dot % (using df_elig)
dots = (
    df_elig[
        (df_elig['legal']==1) &
        (df_elig['score']==0) &
        (df_elig['wide']==0) &
        (df_elig['noball']==0) &
        (df_elig.get('byes',0)==0) &
        (df_elig.get('legbyes',0)==0)
    ]
    .groupby('bowl', as_index=False)['legal'].count()
    .rename(columns={'legal':'dot_count'})
)
agg = agg.merge(dots, on='bowl', how='left').fillna({'dot_count':0})
agg['Dot%'] = 100.0 * agg['dot_count'] / (agg['balls'] + EPS)           # higher better

# Median runs conceded per innings (per bowler)
runs_per_innings = (
    df_elig.groupby(['bowl','p_match','inns'], as_index=False)['runs_bowler'].sum().rename(columns={'runs_bowler':'runs_innings'})
)
median_runs = runs_per_innings.groupby('bowl', as_index=False)['runs_innings'].median().rename(columns={'runs_innings':'median_runs'})
agg = agg.merge(median_runs, on='bowl', how='left').fillna({'median_runs':0})

# Consistency = inverse CV of runs per innings (higher better)
cons = runs_per_innings.groupby('bowl')['runs_innings'].apply(
    lambda x: (x.mean()/(x.std()+EPS)) if x.std() != 0 else 1.0
).reset_index(name='consistency')
agg = agg.merge(cons, on='bowl', how='left').fillna({'consistency':0.0})

# -----------------------
# Percentile computation with correct orientation (100=best)
# -----------------------
# Define which metrics are better when HIGHER
metric_info = {
    'Econ': False,        # lower better
    'SR': False,          # lower better
    'Avg': False,         # lower better
    'Dot%': True,         # higher better
    'Wkts_per_6': True,   # higher better
    'median_runs': False, # lower better (lower median runs conceded is better)
    'consistency': True   # higher better
}

def compute_group_percentiles(df_in, metric, higher_is_better=True, group_col='primary_group'):
    # returns series aligned with df_in.index with percentiles 0..100 where 100 = best
    out = pd.Series(index=df_in.index, dtype=float)
    grouped = df_in.groupby(group_col)
    for g, idx in grouped.groups.items():
        s = df_in.loc[idx, metric]
        N = len(s)
        if N == 1:
            out.loc[idx] = 100.0
            continue
        # ascending ranks: smallest -> 1, largest -> N
        ranks = s.rank(method='average', ascending=True)
        if higher_is_better:
            # larger raw -> higher percentile
            pct = (ranks - 1.0) / (N - 1.0) * 100.0
        else:
            # smaller raw -> higher percentile (invert)
            pct = (N - ranks) / (N - 1.0) * 100.0
        out.loc[idx] = pct
    return out

# compute percentile columns
metric_list = list(metric_info.keys())
for m in metric_list:
    agg[f'percentile_{m}'] = compute_group_percentiles(agg, m, higher_is_better=metric_info[m], group_col='primary_group')

# -----------------------
# PCA weights per primary_group (fallback to equal weights for small groups)
# -----------------------
weights_by_group = {}
percentile_cols = [f'percentile_{m}' for m in metric_list]

for grp in agg['primary_group'].unique():
    grp_df = agg[agg['primary_group'] == grp]
    X = grp_df[percentile_cols].values
    if grp_df.shape[0] > len(metric_list):
        pca = PCA(n_components=1)
        # PCA expects finite numeric matrix
        Xf = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        pca.fit(Xf)
        comp = pca.components_[0]
        # Use absolute component weights (so final score is positive & interpretable)
        w = np.abs(comp)
        if w.sum() == 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / w.sum()
    else:
        # equal weights
        w = np.ones(len(metric_list)) / len(metric_list)
    weights_by_group[grp] = dict(zip(metric_list, w))

# print weights for visibility
print("\nPCA / weight summary (primary_group -> metric weights):")
for grp, ws in weights_by_group.items():
    print(f"  Group: {grp}")
    for m, val in ws.items():
        print(f"    {m}: {val:.3f}")

# -----------------------
# Compute B-ICR (weighted sum of percentiles -> scale 0..100)
# -----------------------
agg['BICR_raw'] = 0.0
for grp, ws in weights_by_group.items():
    mask = agg['primary_group'] == grp
    if mask.sum() == 0:
        continue
    # weighted sum of percentile columns
    vals = np.zeros(mask.sum())
    for m, w in ws.items():
        vals = vals + w * agg.loc[mask, f'percentile_{m}'].values
    agg.loc[mask, 'BICR_raw'] = vals

# BICR is already 0..100 (since percentiles in 0..100 and weights sum to 1)
agg['BICR'] = agg['BICR_raw']

# BICR percentile within each primary_group
agg['BICR_percentile'] = agg.groupby('primary_group')['BICR'].rank(pct=True) * 100.0

# -----------------------
# Final table / sorting
# -----------------------
display_cols = [
    'bowl', 'primary_group', 'phase_groups',
    'balls', 'wickets', 'runs_conceded', 'Econ', 'SR', 'Avg', 'Dot%', 'Wkts_per_6',
    'median_runs', 'consistency',
    'BICR', 'BICR_percentile'
]
final = agg.sort_values(['primary_group', 'BICR_percentile'], ascending=[True, False]).reset_index(drop=True)
print("\nTop bowlers (by primary_group, BICR percentile):")
pd.set_option('display.max_columns', 99)
print(final[display_cols].head(30).to_string(index=False))

# -----------------------
# Show Jasprit Bumrah row if present
# -----------------------
query_name = 'Jasprit Bumrah'
found = final[final['bowl'].str.contains(query_name, case=False, na=False)]
if not found.empty:
    print("\nDetailed row for Jasprit Bumrah (if present):")
    print(found.to_string(index=False))
else:
    print(f"\n'{query_name}' not found among eligible bowlers (or not present in data).")

# End


import pandas as pd
import numpy as np

# parameter for shrinkage strength
K = 5.0

df = final.copy()

# 1) rename originals
if 'BICR' in df.columns:
    df = df.rename(columns={'BICR': 'BICR_initial'})
else:
    df['BICR_initial'] = np.nan

if 'BICR_percentile' in df.columns:
    df = df.rename(columns={'BICR_percentile': 'BICR_percentile_initial'})
else:
    df['BICR_percentile_initial'] = np.nan

# ensure numeric
df['BICR_initial'] = pd.to_numeric(df['BICR_initial'], errors='coerce')
df['match_count'] = pd.to_numeric(df['match_count'], errors='coerce').fillna(0).astype(float)

# 2) global mean
global_mu = df['BICR_initial'].mean(skipna=True)

# 3) shrinkage
def shrink_bicr(row, k=K, mu=global_mu):
    x = row['BICR_initial']
    n = row['match_count']
    if pd.isna(x):
        return np.nan
    if (n is None) or (n <= 0) or pd.isna(n):
        alpha = 0.0
    else:
        alpha = float(n) / (float(n) + float(k))
    return alpha * x + (1.0 - alpha) * mu

df['BICR'] = df.apply(shrink_bicr, axis=1)

# 4) recompute percentiles
def compute_percentile(s):
    non_na = s.dropna()
    n = len(non_na)
    if n == 0:
        return pd.Series([np.nan] * len(s), index=s.index)
    ranks = s.rank(method='average', na_option='keep')
    if n == 1:
        pct = ranks.where(ranks.isna(), 100.0)
    else:
        pct = (ranks - 1.0) / (n - 1.0) * 100.0
    pct = pct.where(~s.isna(), np.nan)
    return pct

df['BICR_percentile'] = compute_percentile(df['BICR'])

# 5) diagnostics
print(f"Shrinkage applied with K={K}. Sample:")
print(df[['bowl', 'match_count', 'BICR_initial', 'BICR',
          'BICR_percentile_initial', 'BICR_percentile']].head(10).to_string(index=False))

# final output
final= df


df=DF

import pandas as pd
import numpy as np

# -----------------------
# Config
# -----------------------
PHASE_LIST = ['Powerplay', 'Middle' ,'Death']
MIN_BALLS = 30         # eligible bowlers must have >= MIN_BALLS legal deliveries
MIN_MATCHES = 3        # player must have >= MIN_MATCHES matches to be included
EPS = 1e-9

# Defensive copy
df = df.copy()

# -----------------------
# Ensure sequential order and delta_wprob present (like your batter code)
# -----------------------
df = df.sort_values(['p_match', 'inns', 'ball_id'])

if 'prev_wprob' not in df.columns:
    df['prev_wprob'] = df.groupby(['p_match', 'inns'])['wprob'].shift(1)
if 'delta_wprob' not in df.columns:
    df['delta_wprob'] = (df['wprob'] - df['prev_wprob']).fillna(0.0)

# Ensure legal column exists and is numeric 0/1
if 'legal' not in df.columns:
    for c in ['noball','wide']:
        if c not in df.columns:
            df[c] = 0
    df['legal'] = (~((df['noball'].astype(bool)) | (df['wide'].astype(bool)))).astype(int)
else:
    df['legal'] = pd.to_numeric(df['legal'], errors='coerce').fillna(0).astype(int)

# Normalize bowler names
df['bowl'] = df['bowl'].astype(str).str.strip()

# -----------------------
# Build runs attributed to bowler per-ball (prefer 'bowlruns' if available)
# -----------------------
if 'bowlruns' in df.columns:
    df['runs_bowler'] = pd.to_numeric(df['bowlruns'], errors='coerce').fillna(0.0)
else:
    # fallback: score minus byes/legbyes (if present)
    df['byes'] = pd.to_numeric(df.get('byes', 0), errors='coerce').fillna(0)
    df['legbyes'] = pd.to_numeric(df.get('legbyes', 0), errors='coerce').fillna(0)
    df['score'] = pd.to_numeric(df.get('score', 0), errors='coerce').fillna(0)
    df['runs_bowler'] = (df['score'] - df['byes'] - df['legbyes']).fillna(0.0)

# -----------------------
# Identify eligible bowlers (>= MIN_BALLS legal deliveries)
# -----------------------
balls_by_bowler = df[df['legal'] == 1].groupby('bowl', as_index=False)['legal'].sum().rename(columns={'legal':'total_legal_balls'})
eligible_bowlers = set(balls_by_bowler.loc[balls_by_bowler['total_legal_balls'] >= MIN_BALLS, 'bowl'].tolist())
print(f"Eligible bowlers (>= {MIN_BALLS} legal balls): {len(eligible_bowlers)}")

if len(eligible_bowlers) == 0:
    raise ValueError("No eligible bowlers found (increase data or lower MIN_BALLS).")

# Work with eligible subset where appropriate
df_elig = df[df['bowl'].isin(eligible_bowlers)].copy()

# -----------------------
# Phase % per bowler (eligible bowlers only)
# -----------------------
phase_counts = (
    df_elig[df_elig['legal'] == 1]
    .groupby(['bowl','phase'], as_index=False)['legal']
    .sum()
    .rename(columns={'legal':'balls_in_phase'})
)

total_balls = phase_counts.groupby('bowl', as_index=False)['balls_in_phase'].sum().rename(columns={'balls_in_phase':'total_balls'})
phase_counts = phase_counts.merge(total_balls, on='bowl', how='left')
phase_counts['phase_pct'] = 100.0 * phase_counts['balls_in_phase'] / (phase_counts['total_balls'] + EPS)

phase_wide = phase_counts.pivot_table(index='bowl', columns='phase', values='phase_pct', fill_value=0.0).reset_index()
for p in PHASE_LIST:
    if p not in phase_wide.columns:
        phase_wide[p] = 0.0

# Phase summary statistics (for printing / inspection)
phase_summary = pd.DataFrame(index=PHASE_LIST)
phase_summary['Min %'] = [phase_wide[p].min() for p in PHASE_LIST]
phase_summary['Max %'] = [phase_wide[p].max() for p in PHASE_LIST]
phase_summary['Avg %'] = [phase_wide[p].mean() for p in PHASE_LIST]
phase_summary['Median %'] = [phase_wide[p].median() for p in PHASE_LIST]
print("\nSummary statistics of phase % across eligible bowlers:")
print(phase_summary.round(2))

phase_medians = phase_summary['Median %'].to_dict()

# -----------------------
# Compute per-match-per-bowler WPA (sum of -delta_wprob on deliveries bowled)
# -----------------------
# We include innings separation since bowling is innings-level; group by p_match+inns+bowl
wpa_per_innings = (
    df_elig
    .groupby(['p_match','inns','bowl'], as_index=False)
    .agg(wpa_innings = ('delta_wprob', lambda s: (-s).sum()))   # negative of batting delta
)

# Now collapse to per-match (if you prefer per-match ignoring innings, sum across inns)
# Keep p_match and bowl as identifiers (mirrors batter code which used p_match+bat)
wpa_per_match = (
    wpa_per_innings
    .groupby(['p_match','bowl'], as_index=False)
    .agg(wpa = ('wpa_innings','sum'))
)

# Compute runs conceded per (p_match,bowl) using final cumulative cur_bowl_runs if available (safer) else sum runs_bowler
if 'cur_bowl_runs' in df_elig.columns:
    # get last row per p_match,bowl (final cumulative)
    last_per_bowler_innings = df_elig.sort_values('ball_id').groupby(['p_match','inns','bowl'], as_index=False).tail(1)
    runs_per_bowler_innings = last_per_bowler_innings.groupby(['p_match','bowl'], as_index=False).agg(runs_conceded_innings=('cur_bowl_runs','max'))
else:
    runs_per_bowler_innings = df_elig.groupby(['p_match','bowl'], as_index=False).agg(runs_conceded_innings=('runs_bowler','sum'))

wpa_per_match = wpa_per_match.merge(runs_per_bowler_innings, on=['p_match','bowl'], how='left').fillna({'runs_conceded_innings':0.0})

# Also compute balls and wickets per p_match,bowl for filtering/metrics
balls_per_match = df_elig.groupby(['p_match','bowl'], as_index=False)['legal'].sum().rename(columns={'legal':'balls_innings'})
wkts_per_match = df_elig.groupby(['p_match','bowl'], as_index=False)['out'].sum().rename(columns={'out':'wkts_innings'})
wpa_per_match = wpa_per_match.merge(balls_per_match, on=['p_match','bowl'], how='left').merge(wkts_per_match, on=['p_match','bowl'], how='left').fillna({'balls_innings':0,'wkts_innings':0})

# -----------------------
# Aggregate per-bowler overall stats (to compute avg_wpa etc.)
# -----------------------
# Only include bowlers with >= MIN_MATCHES matches in wpa_per_match and >= MIN_BALLS overall
match_counts = wpa_per_match.groupby('bowl', as_index=False)['p_match'].nunique().rename(columns={'p_match':'match_count'})
total_balls = df_elig.groupby('bowl', as_index=False)['legal'].sum().rename(columns={'legal':'total_legal_balls'})

# combine and filter
agg_meta = match_counts.merge(total_balls, on='bowl', how='left')
eligible_for_ranking = agg_meta[(agg_meta['match_count'] >= MIN_MATCHES) & (agg_meta['total_legal_balls'] >= MIN_BALLS)]['bowl'].tolist()
print(f"\nBowlers with >= {MIN_MATCHES} matches and >= {MIN_BALLS} balls: {len(eligible_for_ranking)}")

wpa_pm_filt = wpa_per_match[wpa_per_match['bowl'].isin(eligible_for_ranking)].copy()

# compute totals per bowler
player_group_stats = (
    wpa_pm_filt.groupby('bowl', as_index=False)
    .agg(
        total_wpa=('wpa','sum'),
        num_matches=('p_match','nunique'),
        total_runs=('runs_conceded_innings','sum'),
        total_balls = ('balls_innings','sum'),
        total_wkts = ('wkts_innings','sum')
    )
)

player_group_stats['avg_wpa'] = player_group_stats['total_wpa'] / (player_group_stats['num_matches'] + EPS)

# attach phase primary_group and phase_groups from earlier phase_wide logic
# ensure phase_wide covers all eligible bowlers (if a bowler has no phase rows, fill zeros)
phase_for_players = phase_wide[phase_wide['bowl'].isin(player_group_stats['bowl'])].copy()
player_group_stats = player_group_stats.merge(phase_for_players, on='bowl', how='left').fillna(0.0)

# Determine phase_groups and primary_group for these bowlers using same logic as earlier
def derive_groups_for_row(row):
    groups = []
    excess = {}
    for p in PHASE_LIST:
        my_pct = float(row.get(p, 0.0))
        med = float(phase_medians.get(p, 0.0))
        if my_pct >= med - 1e-12:
            groups.append(p)
            excess[p] = my_pct - med
    if len(groups) == 0:
        return pd.Series({'phase_groups': ['Mixed'], 'primary_group': 'Mixed'})
    primary = max(excess.items(), key=lambda kv: kv[1])[0]
    return pd.Series({'phase_groups': groups, 'primary_group': primary})

gp = player_group_stats.apply(derive_groups_for_row, axis=1)
player_group_stats['phase_groups'] = gp['phase_groups']
player_group_stats['primary_group'] = gp['primary_group']

# -----------------------
# Compute percentiles within primary_group (avg_wpa better when HIGHER for bowlers)
# -----------------------
def percentile_within_group(df_in, value_col='avg_wpa', group_col='primary_group'):
    out = pd.Series(index=df_in.index, dtype=float)
    for grp, idx in df_in.groupby(group_col).groups.items():
        s = df_in.loc[idx, value_col]
        N = len(s)
        if N == 1:
            out.loc[idx] = 100.0
            continue
        ranks = s.rank(method='average', ascending=True)
        # For avg_wpa higher is better => larger raw should map to higher percentile
        pct = (ranks - 1.0) / (N - 1.0) * 100.0
        out.loc[idx] = pct
    return out

player_group_stats['percentile_avg_wpa'] = percentile_within_group(player_group_stats, 'avg_wpa', 'primary_group')

# -----------------------
# Sort & display
# -----------------------
player_group_stats = player_group_stats.sort_values(['primary_group','percentile_avg_wpa'], ascending=[True, False]).reset_index(drop=True)

display_cols = ['bowl','primary_group','phase_groups','num_matches','total_balls','total_wkts','total_runs','avg_wpa','percentile_avg_wpa']
pd.set_option('display.max_columns', 99)
print("\nBowler WPA summary (per bowler, aggregated across matches):")
print(player_group_stats[display_cols].head(200).to_string(index=False))

# Show Jasprit Bumrah if present
query_name = 'Jasprit Bumrah'
found = player_group_stats[player_group_stats['bowl'].str.contains(query_name, case=False, na=False)]
if not found.empty:
    print("\nDetailed Jasprit Bumrah row (if present):")
    print(found.to_string(index=False))
else:
    print(f"\n'{query_name}' not found among eligible bowlers (or not present in data).")


player_stats = final

# Paste this function (and its helper functions from your existing module) replacing the prior implementation.
def combine_bowler_scores_phasegroups_master(
    results_df,
    player_stats,
    player_group_stats,
    id_results='bowler',
    id_player_stats='bowl',
    id_group_stats='bowl',
    rvap_col='b_rvap_score',
    icr_col='BICR',
    wpa_col='total_wpa',
    min_matches=MIN_MATCHES_DEFAULT,
    prior_weights=None,
    alpha=DEFAULT_ALPHA,
    method_primary='reliability',
    outcome_col='avg_wpa',
    winsor_pct=WINSOR_PCT,
    # NEW params for consistency bonus
    consistency_coefficient=20.0,
    min_balls_threshold=DEFAULT_MIN_BALLS_THRESHOLD,
    # safety control: divide coefficient by this to avoid huge jumps (tweakable)
    consistency_multiplier_divider=5.0
):
    """
    Combine results_df (rvap), player_stats (BICR-like), and player_group_stats (WPA)
    and compute final ICR percentiles.

    Adds a log-based consistency bonus safely:
      raw_bonus = log1p(total_legal_balls) / log1p(min_balls_threshold)
      bonus_z  = standardize(raw_bonus) within comp_group (mean0,std1), clipped to [-3,3]
      applied_term = (consistency_coefficient / consistency_multiplier_divider) * bonus_z
      icr_z_final += applied_term
    """
    if prior_weights is None:
        prior_weights = DEFAULT_PRIOR.copy()

    # Defensive copies
    r = results_df.copy()
    ps = player_stats.copy()
    pg = player_group_stats.copy()

    # Validate id columns exist
    for df_obj, col, name in [(r, id_results, 'results_df'), (ps, id_player_stats, 'player_stats'), (pg, id_group_stats, 'player_group_stats')]:
        if col not in df_obj.columns:
            raise KeyError(f"{name} missing id column '{col}'")

    # normalize IDs (lowercase/trim)
    r[id_results] = r[id_results].astype(str).str.strip().str.lower()
    ps[id_player_stats] = ps[id_player_stats].astype(str).str.strip().str.lower()
    pg[id_group_stats] = pg[id_group_stats].astype(str).str.strip().str.lower()

    # rename to a canonical 'player'
    r = r.rename(columns={id_results: 'player'})
    ps = ps.rename(columns={id_player_stats: 'player'})
    pg = pg.rename(columns={id_group_stats: 'player'})

    # parse phase_groups -> comp_group_candidate on all three tables
    for df_obj in (r, ps, pg):
        if 'phase_groups' in df_obj.columns:
            df_obj['phase_groups_parsed'] = df_obj['phase_groups'].apply(_extract_primary_group)
        else:
            df_obj['phase_groups_parsed'] = 'Other'
        df_obj['comp_group_candidate'] = df_obj['phase_groups_parsed'].fillna('Other').astype(str).str.strip()

    # take results_df comp_group as authoritative
    r['comp_group'] = r['comp_group_candidate']

    # build common-player intersection
    players_r = set(r['player'].dropna().unique())
    players_ps = set(ps['player'].dropna().unique())
    players_pg = set(pg['player'].dropna().unique())
    common_players = players_r & players_ps & players_pg
    if len(common_players) == 0:
        raise ValueError("No common bowlers found in all three datasets after ID normalization.")

    # filter to common players
    r_f = r[r['player'].isin(common_players)].copy()
    ps_f = ps[ps['player'].isin(common_players)].copy()
    pg_f = pg[pg['player'].isin(common_players)].copy()

    # enforce min_matches on player_group_stats (num_matches)
    if 'num_matches' in pg_f.columns:
        pg_ok = pg_f[pg_f['num_matches'] >= min_matches].copy()
        if pg_ok.empty:
            pg_ok = pg_f[pg_f['num_matches'] >= MIN_MATCHES_FALLBACK].copy()
    else:
        pg_ok = pg_f.copy()

    eligible_players = set(pg_ok['player'].unique()) & common_players
    if len(eligible_players) == 0:
        raise ValueError("No eligible bowlers after applying min_matches filter on player_group_stats.")

    # final filtered tables
    r_f = r_f[r_f['player'].isin(eligible_players)].copy()
    ps_f = ps_f[ps_f['player'].isin(eligible_players)].copy()
    pg_f = pg_f[pg_f['player'].isin(eligible_players)].copy()

    # select needed columns and rename metrics to canonical names
    merge1_cols = ['player', 'comp_group'] + ([rvap_col] if rvap_col in r_f.columns else [])
    merge1 = r_f[merge1_cols].copy().rename(columns={rvap_col: 'rvap_raw'})

    merge2_cols = ['player'] + ([icr_col] if icr_col in ps_f.columns else [])
    merge2 = ps_f[merge2_cols].copy().rename(columns={icr_col: 'icr_raw'})

    merge3_cols = ['player'] + ([wpa_col] if wpa_col in pg_f.columns else [])
    merge3 = pg_f[merge3_cols].copy().rename(columns={wpa_col: 'wpa_raw'})

    # inner join on player only
    merged = merge1.merge(merge2, on='player', how='inner').merge(merge3, on='player', how='inner')
    if merged.empty:
        raise ValueError("Merged dataframe is empty after joining on player. Check IDs and that metrics exist in each table.")

    # prepare counts n1,n2,n3 from available columns (fall back to 1)
    merged['n1'] = pd.to_numeric(r_f.set_index('player').reindex(merged['player']).get('total_innings', 1), errors='coerce').fillna(1.0).values
    merged['n2'] = pd.to_numeric(ps_f.set_index('player').reindex(merged['player']).get('match_count', 1), errors='coerce').fillna(1.0).values
    merged['n3'] = pd.to_numeric(pg_f.set_index('player').reindex(merged['player']).get('num_matches', 1), errors='coerce').fillna(1.0).values

    # ensure numeric metrics
    for c in ['rvap_raw', 'icr_raw', 'wpa_raw']:
        merged[c] = pd.to_numeric(merged.get(c, np.nan), errors='coerce')

    # drop rows with all three metrics missing
    merged = merged.dropna(subset=['rvap_raw','icr_raw','wpa_raw'], how='all').reset_index(drop=True)

    # winsorize and z-score within comp_group
    merged[['rvap_w','icr_w','wpa_w']] = merged.groupby('comp_group')[['rvap_raw','icr_raw','wpa_raw']].transform(
        lambda x: _winsorize_series(x, pct=min(winsor_pct, 1/len(x)) if len(x) > 5 else 0)
    )
    merged[['rvap_z','icr_z','wpa_z']] = merged.groupby('comp_group')[['rvap_w','icr_w','wpa_w']].transform(_standardize_series)

    # compute group-level weights and blended icr_z_final
    merged['icr_z_final'] = np.nan
    merged['icr_percentile_final'] = np.nan
    weight_summary = {}

    for group in merged['comp_group'].unique():
        mask = merged['comp_group'] == group
        g = merged[mask].copy()
        group_size = len(g)

        if group_size < MIN_GROUP_SIZE:
            data_w_rel = data_w_inv = data_w_pca = data_w_reg = [1/3,1/3,1/3]
            pca_var = reg_r2 = 0.0
        else:
            rel_tuple = compute_weights_reliability(g['n1'].values, g['n2'].values, g['n3'].values)
            data_w_rel = (np.nanmean(rel_tuple[0]), np.nanmean(rel_tuple[1]), np.nanmean(rel_tuple[2]))
            data_w_inv = compute_weights_invvar(g['rvap_z'].values, g['icr_z'].values, g['wpa_z'].values,
                                               np.nanmean(g['n1']), np.nanmean(g['n2']), np.nanmean(g['n3']))
            data_w_pca, pca_var = compute_weights_pca(g['rvap_z'].values, g['icr_z'].values, g['wpa_z'].values)
            data_w_reg, reg_r2 = compute_weights_regression(g['rvap_z'].values, g['icr_z'].values, g['wpa_z'].values,
                                                            g[outcome_col].values if outcome_col in g.columns else g['wpa_raw'].values)

        data_w = {
            'reliability': data_w_rel,
            'invvar': data_w_inv,
            'pca': data_w_pca,
            'regression': data_w_reg
        }.get(method_primary, data_w_rel)

        final_w_raw = (
            alpha * prior_weights['rvap'] + (1 - alpha) * data_w[0],
            alpha * prior_weights['icr']  + (1 - alpha) * data_w[1],
            alpha * prior_weights['wpa']  + (1 - alpha) * data_w[2]
        )
        s = sum(final_w_raw)
        final_w = [x / s for x in final_w_raw] if s != 0 else [1/3,1/3,1/3]

        # blended z
        merged.loc[mask, 'icr_z_final'] = (final_w[0] * merged.loc[mask, 'rvap_z'] +
                                          final_w[1] * merged.loc[mask, 'icr_z'] +
                                          final_w[2] * merged.loc[mask, 'wpa_z'])

        # ---------- SAFELY apply consistency log bonus ----------
        # get total_legal_balls for these players (from results r_f). fallback to 0 if missing
        total_balls_series = pd.to_numeric(r_f.set_index('player').reindex(merged.loc[mask,'player']).get('total_legal_balls', 0), errors='coerce').fillna(0).astype(float)

        # raw bonus (same as you asked)
        denom = np.log1p(float(max(1.0, min_balls_threshold)))
        raw_bonus = np.log1p(total_balls_series) / denom

        # standardize the raw_bonus within the group (mean 0, std 1)
        mean_raw = raw_bonus.mean() if len(raw_bonus) > 0 else 0.0
        std_raw = raw_bonus.std(ddof=1) if len(raw_bonus) > 1 else 0.0
        if std_raw <= 1e-9:
            bonus_z = (raw_bonus - mean_raw).fillna(0.0)  # all zeros effectively
        else:
            bonus_z = (raw_bonus - mean_raw) / std_raw

        # clip extreme z-values (safety)
        bonus_z_clipped = bonus_z.clip(-3.0, 3.0)

        # apply multiplier but scaled down to be safe; user can tweak divider
        applied_term = (consistency_coefficient / float(max(1.0, consistency_multiplier_divider))) * bonus_z_clipped.values

        merged.loc[mask, 'icr_z_final'] = merged.loc[mask, 'icr_z_final'].fillna(0.0).values + applied_term

        # ---------- compute percentile after bonus ----------
        merged.loc[mask, 'icr_percentile_final'] = norm.cdf(merged.loc[mask, 'icr_z_final']) * 100.0

        weight_summary[group] = {
            'final_weights': {'rvap': final_w[0], 'icr': final_w[1], 'wpa': final_w[2]},
            'data_weights': {'reliability': data_w_rel, 'invvar': data_w_inv, 'pca': data_w_pca, 'regression': data_w_reg},
            'pca_var': pca_var if 'pca_var' in locals() else 0.0,
            'reg_r2': reg_r2 if 'reg_r2' in locals() else 0.0,
            'group_size': group_size,
            'consistency_coefficient': consistency_coefficient,
            'min_balls_threshold': min_balls_threshold,
            'consistency_multiplier_divider': consistency_multiplier_divider
        }

    # diagnostics
    merged['rvap_score_percentile'] = merged.groupby('comp_group')['rvap_raw'].rank(pct=True) * 100
    merged['icr_rank_final'] = merged.groupby('comp_group')['icr_percentile_final'].rank(method='min', ascending=False)
    corr_wpa = spearmanr(merged['icr_percentile_final'], merged['wpa_raw'], nan_policy='omit')[0]
    corr_rvap = spearmanr(merged['icr_percentile_final'], merged['rvap_raw'], nan_policy='omit')[0]
    weight_summary['validation'] = {'spearman_corr_with_wpa': corr_wpa, 'spearman_corr_with_rvap': corr_rvap}

    merged = merged.sort_values(['comp_group', 'icr_percentile_final'], ascending=[True, False]).reset_index(drop=True)
    return merged, weight_summary

merged, weights = combine_bowler_scores_phasegroups_master(
    results_df, player_stats, player_group_stats,
    id_results='bowler', id_player_stats='bowl', id_group_stats='bowl',
    rvap_col='b_rvap_score', icr_col='BICR', wpa_col='total_wpa',
    min_matches=5, prior_weights={'rvap':0.2,'icr':0.5,'wpa':0.3},
    alpha=0.6, method_primary='reliability', outcome_col='avg_wpa',
    winsor_pct=0.01, consistency_coefficient=3.0, min_balls_threshold=30
)
print(merged.head())


import pandas as pd
import numpy as np

# parameter for shrinkage strength
K = 5.0

df = merged.copy()

# 1) rename originals
if 'icr_z_final' in df.columns:
    df = df.rename(columns={'icr_z_final': 'icr_z_final_initial'})
else:
    df['icr_z_final_initial'] = np.nan

if 'icr_percentile_final' in df.columns:
    df = df.rename(columns={'icr_percentile_final': 'icr_percentile_final_initial'})
else:
    df['icr_percentile_final_initial'] = np.nan

# ensure numeric
df['icr_z_final_initial'] = pd.to_numeric(df['icr_z_final_initial'], errors='coerce')
df['n1'] = pd.to_numeric(df['n1'], errors='coerce').fillna(0).astype(float)

# 2) global mean (since no group info here)
global_mu = df['icr_z_final_initial'].mean(skipna=True)

# 3) shrinkage
def shrink_icr(row, k=K, mu=global_mu):
    x = row['icr_z_final_initial']
    n = row['n1']
    if pd.isna(x):
        return np.nan
    if (n is None) or (n <= 0) or pd.isna(n):
        alpha = 0.0
    else:
        alpha = float(n) / (float(n) + float(k))
    return alpha * x + (1.0 - alpha) * mu

df['icr_z_final'] = df.apply(shrink_icr, axis=1)

# 4) recompute percentiles (global, not group-based)
def compute_percentile(s):
    non_na = s.dropna()
    n = len(non_na)
    if n == 0:
        return pd.Series([np.nan] * len(s), index=s.index)
    ranks = s.rank(method='average', na_option='keep')
    if n == 1:
        pct = ranks.where(ranks.isna(), 100.0)
    else:
        pct = (ranks - 1.0) / (n - 1.0) * 100.0
    pct = pct.where(~s.isna(), np.nan)
    return pct

df['icr_percentile_final'] = compute_percentile(df['icr_z_final'])

# 5) diagnostics
print(f"Shrinkage applied with K={K}. Sample:")
print(df[['player', 'n1', 'icr_z_final_initial', 'icr_z_final',
          'icr_percentile_final_initial', 'icr_percentile_final']].head(10).to_string(index=False))

# final output
merged = df
df=DF

merged=merged.sort_values('icr_percentile_final', ascending = False)

icr_df=merged

icr_df['bowler'] = icr_df['player'].astype(str).str.strip().str.title().replace({'nan': np.nan})

# Ensure lowercase for consistent merge
icr_df['player_lower'] = icr_df['player'].str.lower()
df['bowl_lower'] = df['bowl'].str.lower()

# Merge on lowercase version
icr_df = icr_df.merge(df[['bowl_lower', 'team_bowl']],
                      left_on='player_lower', right_on='bowl_lower',
                      how='left')

# Drop helper column if not needed
icr_df = icr_df.drop(columns=['player_lower', 'bowl_lower'])


icr_df=icr_df.drop_duplicates(subset=['player'])

icr_df[['bowler','team_bowl','icr_percentile_final']].head(20)

bicr_25=icr_df

