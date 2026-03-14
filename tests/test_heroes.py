"""Tests for hero list loading, edit distance, and fuzzy matching."""


from overwatchlooker.heroes import ALL_HEROES, edit_distance, match_hero_name


class TestHeroList:
    def test_heroes_loads_nonempty(self):
        assert len(ALL_HEROES) > 0

    def test_heroes_contains_known(self):
        for hero in ["Ana", "Reinhardt", "D.Va", "Soldier: 76", "Wrecking Ball"]:
            assert hero in ALL_HEROES, f"{hero} missing from ALL_HEROES"

    def test_heroes_no_duplicates(self):
        assert len(ALL_HEROES) == len(set(ALL_HEROES))

    def test_heroes_alphabetical(self):
        """Heroes list should be roughly alphabetical (warn if not)."""
        # Not a hard requirement, just check it's not randomly shuffled
        # by verifying the first and last entries are in expected range
        assert ALL_HEROES[0][0] in "AB"  # starts with A or B
        assert ALL_HEROES[-1][0] in "YZ"  # ends with Y or Z

    def test_heroes_no_blank_entries(self):
        for hero in ALL_HEROES:
            assert hero.strip() == hero
            assert len(hero) > 0


class TestEditDistance:
    def test_identical(self):
        assert edit_distance("ana", "ana") == 0

    def test_one_insertion(self):
        assert edit_distance("ana", "anna") == 1

    def test_one_deletion(self):
        assert edit_distance("reinhardt", "reinhadt") == 1

    def test_one_substitution(self):
        assert edit_distance("juno", "jnno") == 1

    def test_completely_different(self):
        d = edit_distance("abc", "xyz")
        assert d == 3

    def test_symmetric(self):
        assert edit_distance("mercy", "merci") == edit_distance("merci", "mercy")

    def test_empty_strings(self):
        assert edit_distance("", "") == 0
        assert edit_distance("abc", "") == 3
        assert edit_distance("", "abc") == 3

    def test_case_sensitive(self):
        assert edit_distance("Ana", "ana") == 1


class TestMatchHeroName:
    def test_exact_match(self):
        assert match_hero_name("Reinhardt") == "Reinhardt"

    def test_case_insensitive(self):
        assert match_hero_name("reinhardt") == "Reinhardt"

    def test_all_caps(self):
        assert match_hero_name("REINHARDT") == "Reinhardt"

    def test_ocr_trailing_noise(self):
        assert match_hero_name("Reinhardtg") == "Reinhardt"

    def test_soldier_76(self):
        # "Soldier 76" should match "Soldier: 76" (spaces stripped in comparison)
        assert match_hero_name("Soldier76") == "Soldier: 76"

    def test_dva(self):
        assert match_hero_name("DVa") == "D.Va"

    def test_wrecking_ball(self):
        assert match_hero_name("WreckingBall") == "Wrecking Ball"

    def test_garbage_no_match(self):
        assert match_hero_name("xyzzy123blah") == ""

    def test_empty_string(self):
        assert match_hero_name("") == ""

    def test_whitespace_only(self):
        assert match_hero_name("   ") == ""

    def test_short_hero_name(self):
        assert match_hero_name("Ana") == "Ana"
        assert match_hero_name("Mei") == "Mei"

    def test_partial_prefix(self):
        # "Rein" is 4 chars, "Reinhardt" is 9 → edit distance 5
        # Threshold is max(2, 9*0.4)=3.6 → won't match, which is correct
        # But "Reinha" → distance 3 → matches
        result = match_hero_name("Reinha")
        assert result == "Reinhardt"
