from creditbot import privacy


def test_identifiers_are_removed():
    text = "my name is Ravi Kumar, pan ABCDE1234F, aadhaar 1234 5678 9012, call 9876543210 or ravi@x.com"
    out = privacy.mask_text(text)
    for secret in ("Ravi", "ABCDE1234F", "5678", "9876543210", "ravi@x.com"):
        assert secret not in out


def test_pincode_and_phone_masks():
    assert privacy.mask_pincode("560002") == "560***"
    assert privacy.mask_phone("9876543210") == "98******10"
    assert privacy.mask_answer("pincode", "560002") == "560***"


def test_free_text_pincode_cut_in_logs_but_kept_for_llm():
    assert "560002" not in privacy.mask_text("i live at 560002")
    assert "560002" in privacy.mask_for_llm("i live at 560002")


def test_tokens_are_stable_and_keyed():
    a, b = privacy.token("x", "one"), privacy.token("x", "two")
    assert a == privacy.token("x", "one") and a != b
