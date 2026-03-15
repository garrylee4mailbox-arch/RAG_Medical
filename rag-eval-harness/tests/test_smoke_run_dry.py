def test_smoke_run_limit0():
    from src.eval.run_eval import main
    main(["--run","smoke","--limit","0"])
