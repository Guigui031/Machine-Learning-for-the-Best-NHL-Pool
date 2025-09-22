"""
Hockey-Specific Feature Engineering
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class HockeyFeatures:
    """Hockey-specific feature engineering methods."""

    @staticmethod
    def create_streak_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on performance streaks.

        Args:
            df: Input DataFrame with player data

        Returns:
            DataFrame with streak features added
        """
        logger.info("Creating streak features...")
        df_streak = df.copy()

        # Simulate streak data (in real implementation, you'd have game-by-game data)
        # For now, we'll estimate based on consistency metrics

        for season in ['1', '2']:
            # Simulate hot/cold streaks based on performance variance
            goals_col = f'goals_{season}'
            assists_col = f'assists_{season}'
            games_col = f'games_{season}'

            if all(col in df_streak.columns for col in [goals_col, assists_col, games_col]):
                # Estimate performance consistency as proxy for streakiness
                total_points = df_streak[goals_col] + df_streak[assists_col]
                avg_ppg = total_points / df_streak[games_col].replace(0, np.nan)

                # Simulate streak metrics (replace with actual game-by-game data)
                df_streak[f'avg_streak_length_{season}'] = np.random.normal(3, 1, len(df_streak))
                df_streak[f'max_point_streak_{season}'] = avg_ppg * np.random.uniform(2, 6, len(df_streak))
                df_streak[f'max_goal_streak_{season}'] = (df_streak[goals_col] / df_streak[games_col].replace(0, np.nan)) * np.random.uniform(2, 4, len(df_streak))

        logger.info("Streak features created (simulated)")
        return df_streak

    @staticmethod
    def create_team_context_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on team context and performance.

        Args:
            df: Input DataFrame with player data

        Returns:
            DataFrame with team context features added
        """
        logger.info("Creating team context features...")
        df_team = df.copy()

        # Team performance proxies
        for season in ['1', '2']:
            team_col = f'team_{season}'
            if team_col in df_team.columns:
                # Team strength (estimated from average team performance)
                team_stats = df_team.groupby(team_col).agg({
                    f'goals_{season}': 'mean',
                    f'assists_{season}': 'mean',
                    f'games_{season}': 'mean'
                }).reset_index()

                team_stats[f'team_avg_ppg_{season}'] = (
                    (team_stats[f'goals_{season}'] + team_stats[f'assists_{season}']) /
                    team_stats[f'games_{season}'].replace(0, np.nan)
                )

                # Merge back to main dataframe
                df_team = df_team.merge(
                    team_stats[[team_col, f'team_avg_ppg_{season}']],
                    on=team_col,
                    how='left'
                )

                # Player's performance relative to team
                player_ppg_col = f'ppg_{season}'
                if player_ppg_col in df_team.columns:
                    df_team[f'player_vs_team_performance_{season}'] = (
                        df_team[player_ppg_col] - df_team[f'team_avg_ppg_{season}']
                    )

        # Team changes
        if all(col in df_team.columns for col in ['team_1', 'team_2']):
            df_team['changed_teams'] = (df_team['team_1'] != df_team['team_2']).astype(int)

        logger.info("Team context features created")
        return df_team

    @staticmethod
    def create_age_curve_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create age curve features specific to hockey positions.

        Args:
            df: Input DataFrame with player data

        Returns:
            DataFrame with age curve features added
        """
        logger.info("Creating age curve features...")
        df_age = df.copy()

        if 'age' not in df_age.columns or 'role' not in df_age.columns:
            logger.warning("Missing 'age' or 'role' columns for age curve features")
            return df_age

        # Position-specific age curves
        def get_age_factor(age, position):
            """Get age factor based on position-specific curves."""
            if position == 'A':  # Attackers peak around 27-28
                if age < 23:
                    return 0.85  # Still developing
                elif age < 28:
                    return 1.0   # Peak years
                elif age < 32:
                    return 0.95  # Slight decline
                else:
                    return 0.8   # Veteran decline
            elif position == 'D':  # Defensemen peak later, around 28-30
                if age < 24:
                    return 0.8
                elif age < 30:
                    return 1.0
                elif age < 34:
                    return 0.95
                else:
                    return 0.75
            elif position == 'G':  # Goalies peak even later
                if age < 25:
                    return 0.85
                elif age < 32:
                    return 1.0
                elif age < 36:
                    return 0.9
                else:
                    return 0.7
            else:
                return 1.0

        df_age['age_factor'] = df_age.apply(
            lambda row: get_age_factor(row['age'], row['role']), axis=1
        )

        # Distance from peak age by position
        peak_ages = {'A': 27, 'D': 29, 'G': 30}
        df_age['years_from_peak'] = df_age.apply(
            lambda row: abs(row['age'] - peak_ages.get(row['role'], 27)), axis=1
        )

        # Career stage indicators
        df_age['is_rookie'] = (df_age['age'] <= 22).astype(int)
        df_age['is_peak_age'] = ((df_age['age'] >= 25) & (df_age['age'] <= 30)).astype(int)
        df_age['is_veteran'] = (df_age['age'] >= 33).astype(int)

        logger.info("Age curve features created")
        return df_age

    @staticmethod
    def create_injury_risk_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create features related to injury risk and durability.

        Args:
            df: Input DataFrame with player data

        Returns:
            DataFrame with injury risk features added
        """
        logger.info("Creating injury risk features...")
        df_injury = df.copy()

        # Games played consistency as durability indicator
        if all(col in df_injury.columns for col in ['games_1', 'games_2']):
            df_injury['durability_score'] = (df_injury['games_1'] + df_injury['games_2']) / (2 * 82)
            df_injury['games_variance'] = abs(df_injury['games_1'] - df_injury['games_2'])
            df_injury['injury_risk_indicator'] = (df_injury['games_variance'] > 20).astype(int)

        # Age-related injury risk
        if 'age' in df_injury.columns:
            df_injury['age_injury_risk'] = np.where(
                df_injury['age'] > 30,
                (df_injury['age'] - 30) * 0.1,  # Increased risk after 30
                0
            )

        # Position-specific injury risk
        if 'role' in df_injury.columns:
            # Forwards generally have higher injury rates than goalies
            position_risk = {'A': 1.2, 'D': 1.1, 'G': 0.8}
            df_injury['position_injury_risk'] = df_injury['role'].map(position_risk).fillna(1.0)

        # Combined injury risk score
        risk_columns = ['age_injury_risk', 'position_injury_risk']
        if all(col in df_injury.columns for col in risk_columns):
            df_injury['total_injury_risk'] = df_injury[risk_columns].sum(axis=1)

        logger.info("Injury risk features created")
        return df_injury

    @staticmethod
    def create_performance_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create features capturing performance momentum and trends.

        Args:
            df: Input DataFrame with player data

        Returns:
            DataFrame with momentum features added
        """
        logger.info("Creating performance momentum features...")
        df_momentum = df.copy()

        # Performance trajectory
        for stat in ['goals', 'assists']:
            stat1_col = f'{stat}_1'
            stat2_col = f'{stat}_2'
            if all(col in df_momentum.columns for col in [stat1_col, stat2_col]):
                # Year-over-year change
                df_momentum[f'{stat}_yoy_change'] = df_momentum[stat2_col] - df_momentum[stat1_col]

                # Percentage change
                df_momentum[f'{stat}_yoy_pct_change'] = (
                    (df_momentum[stat2_col] - df_momentum[stat1_col]) /
                    df_momentum[stat1_col].replace(0, np.nan) * 100
                )

                # Momentum category (convert to string to avoid categorical issues)
                df_momentum[f'{stat}_momentum'] = pd.cut(
                    df_momentum[f'{stat}_yoy_pct_change'],
                    bins=[-np.inf, -20, -5, 5, 20, np.inf],
                    labels=['declining', 'slight_decline', 'stable', 'improving', 'breakout']
                ).astype(str)

        # Overall performance momentum
        if all(col in df_momentum.columns for col in ['ppg_1', 'ppg_2']):
            df_momentum['performance_momentum'] = (
                df_momentum['ppg_2'] - df_momentum['ppg_1']
            ) / df_momentum['ppg_1'].replace(0, np.nan)

            # Momentum strength categories (convert to string to avoid categorical issues)
            df_momentum['momentum_strength'] = pd.cut(
                df_momentum['performance_momentum'],
                bins=[-np.inf, -0.2, -0.05, 0.05, 0.2, np.inf],
                labels=['strong_decline', 'decline', 'stable', 'improvement', 'strong_improvement']
            ).astype(str)

        logger.info("Performance momentum features created")
        return df_momentum

    @staticmethod
    def create_all_hockey_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create all hockey-specific features.

        Args:
            df: Input DataFrame with player data

        Returns:
            DataFrame with all hockey features added
        """
        logger.info("Creating all hockey-specific features...")

        df_hockey = df.copy()

        # Apply all hockey feature engineering methods
        df_hockey = HockeyFeatures.create_streak_features(df_hockey)
        df_hockey = HockeyFeatures.create_team_context_features(df_hockey)
        df_hockey = HockeyFeatures.create_age_curve_features(df_hockey)
        df_hockey = HockeyFeatures.create_injury_risk_features(df_hockey)
        df_hockey = HockeyFeatures.create_performance_momentum_features(df_hockey)

        logger.info(f"All hockey features created. Total columns: {len(df_hockey.columns)}")
        return df_hockey

    @staticmethod
    def get_feature_groups() -> Dict[str, List[str]]:
        """Get logical groupings of hockey features.

        Returns:
            Dictionary mapping feature group names to feature name patterns
        """
        return {
            'basic_stats': ['goals_', 'assists_', 'points_', 'games_', 'ppg_'],
            'advanced_stats': ['shooting_pct_', 'toi_per_game_', 'plus_minus_'],
            'streaks': ['streak_', 'max_point_streak_', 'max_goal_streak_'],
            'team_context': ['team_avg_', 'player_vs_team_', 'changed_teams'],
            'age_curves': ['age_factor', 'years_from_peak', 'is_rookie', 'is_peak_age', 'is_veteran'],
            'injury_risk': ['durability_', 'injury_risk_', 'games_variance'],
            'momentum': ['yoy_change', 'yoy_pct_change', 'momentum', 'performance_momentum'],
            'physical': ['height', 'weight', 'bmi'],
            'position': ['role_', 'is_attacker', 'is_defenseman', 'is_goalie'],
            'consistency': ['consistency', 'games_consistency']
        }