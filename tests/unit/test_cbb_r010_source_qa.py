import importlib.util
from pathlib import Path
import unittest
import tempfile

import pandas as pd

script_path = Path(__file__).resolve().parents[2] / 'scripts' / 'cbb_r010_source_qa.py'
if not script_path.exists():
    script_path = Path(__file__).with_name('cbb_r010_source_qa.py')
spec = importlib.util.spec_from_file_location('r010', str(script_path))
r010 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r010)


class TestR010(unittest.TestCase):
    def fixtures(self):
        return {
            'ncaa_possessions': pd.DataFrame([
                {'contest_id': 'N1', 'espn_game_id': 123, 'period': 1, 'poss_team_espn_team_id': 10, 'poss_num': 1, 'pts': 2},
                {'contest_id': 'N1', 'espn_game_id': 123, 'period': 1, 'poss_team_espn_team_id': 10, 'poss_num': 2, 'pts': 0},
            ]),
            'ncaa_player_box': pd.DataFrame([
                {'contest_id': 'N1', 'espn_game_id': 123, 'team_espn_team_id': 10, 'player_id': 'ncaa1', 'clean_name': 'John Doe', 'mins': 30, 'pts': 17},
                {'contest_id': 'N1', 'espn_game_id': 123, 'team_espn_team_id': 10, 'player_id': 'ncaa2', 'clean_name': 'Smith Lee', 'mins': 10, 'pts': 5}
            ]),
            'espn_game_rosters': pd.DataFrame([
                {'game_id': 123.0, 'team_id': '10', 'athlete_id': 'e1', 'athlete_display_name': 'John Doe', 'active': True, 'did_not_play': False},
                {'game_id': 123.0, 'team_id': '10', 'athlete_id': 'e2', 'athlete_display_name': 'Lee Smith', 'active': True, 'did_not_play': False}
            ]),
            'espn_season_rosters': pd.DataFrame([
                {'season': 2027, 'team_id': 10, 'athlete_id': 'e1'},
                {'season': 2027, 'team_id': 10, 'athlete_id': 'e2'}
            ])
        }

    def test_schema_and_conservative_player_join(self):
        sums, errs, games, cross = r010.qa_sources(self.fixtures())
        self.assertTrue(sums.schema_pass.all())
        self.assertEqual(len(games), 1)
        self.assertEqual(int(games.iloc[0].event_possessions), 2)
        self.assertEqual(cross.match_status.tolist(), ['MATCH_EXACT_GAME_TEAM_NAME', 'UNMATCHED'])
        self.assertTrue((cross.feature_availability == 'POSTGAME_DIAGNOSTIC').all())

    def test_duplicate_and_active_dnp(self):
        f = self.fixtures()
        f['espn_game_rosters'] = pd.concat([f['espn_game_rosters'], f['espn_game_rosters'].iloc[[0]]], ignore_index=True)
        f['espn_game_rosters'].loc[0,'did_not_play'] = True
        sums, errs, games, cross = r010.qa_sources(f)
        self.assertIn('DUPLICATE_NATIVE_PLAYER', set(errs.kind))
        self.assertIn('ACTIVE_DNP_CONFLICT', set(errs.kind))
        self.assertEqual(cross.iloc[0].match_status, 'AMBIGUOUS_DUPLICATE_ROW')

    def test_returning_minutes_never_guess(self):
        prior = pd.DataFrame([
            {'team_id': 10, 'athlete_id': 'e1', 'minutes': 120},
            {'team_id': 10, 'athlete_id': 'e2', 'minutes': 80}
        ])
        current = pd.DataFrame([{'team_id': 10, 'athlete_id': 'e1'}])
        g = r010.returning_minutes(prior, current)
        self.assertAlmostEqual(float(g.iloc[0].returning_minutes_share), 0.6)
        self.assertEqual(g.iloc[0].data_status, 'SOURCE_PIT_REVIEW_REQUIRED')
        prior.loc[1,'minutes'] = None
        with self.assertRaises(ValueError):
            r010.returning_minutes(prior, current)

    def test_duplicate_possession_is_flagged(self):
        f = self.fixtures()
        f['ncaa_possessions'] = pd.concat([f['ncaa_possessions'], f['ncaa_possessions'].iloc[[0]]], ignore_index=True)
        _, errs, _, _ = r010.qa_sources(f)
        self.assertIn('DUPLICATE_POSS_NUM', set(errs.kind))

    def test_fetch_never_overwrites_existing_local_file(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'a.parquet'
            p.write_bytes(b'pre-existing-data')
            got = r010.safe_fetch('https://example.invalid/never-retrieve', p)
            self.assertEqual(got['status'], 'ALREADY_PRESENT')
            self.assertEqual(p.read_bytes(), b'pre-existing-data')

    def test_schema_gate(self):
        f = self.fixtures()
        del f['ncaa_possessions']['contest_id']
        sums, errs, games, cross = r010.qa_sources(f)
        self.assertFalse(sums.schema_pass.all())
        self.assertEqual(len(games), 0)
        self.assertIn('SCHEMA', set(errs.kind))


if __name__ == '__main__':
    unittest.main()