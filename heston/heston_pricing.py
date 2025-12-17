import QuantLib as ql
import numpy as np

def price_heston_american_option(S, K, r, q, T, v0,
                                 theta, kappa, sigma, rho,
                                 todays_date, calendar, day_count,
                                 option_type = "call", engine_grid=(600, 2000, 900), damping_steps=10):
    """
    Prices a single American option (call or put) with the Heston model.

    Parameters
    ----------
    option_type : {"call","put","c","p"} (case-insensitive)
        Type of option to price.

    Returns
    -------
    float
        The American option price (NPV). np.nan if QuantLib throws a RuntimeError.
    """

    # 0) set QuantLib option type
    opt_str = option_type.lower()
    if opt_str in ["call", "c"]:
        ql_opt_type = ql.Option.Call
    elif opt_str in ["put", "p"]:
        ql_opt_type = ql.Option.Put
    else:
        raise ValueError(f"Invalid option_type: {option_type}. Use 'call' or 'put'.")
    
    # 1) Build expiry date from T (in years)

    # anchor QL to today's date
    ql.Settings.instance().evaluationDate = todays_date

    days = int(max(1, round(float(T) * 365.0)))
    expiry_date = calendar.advance(todays_date,
                                   ql.Period(days, ql.Days))

    # 2) Flat curves
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(float(S)))
    rf_curve = ql.YieldTermStructureHandle(ql.FlatForward(todays_date, float(r), day_count))
    div_curve = ql.YieldTermStructureHandle(ql.FlatForward(todays_date, float(q), day_count))

    # 3) Heston process & model
    heston_process = ql.HestonProcess(
        rf_curve, div_curve, spot_handle,
        float(v0), float(kappa), float(theta), float(sigma), float(rho)
    )
    heston_model = ql.HestonModel(heston_process)

    # 4) Engine settings
    t_grid, x_grid, v_grid = engine_grid
    scheme_desc = ql.FdmSchemeDesc.Hundsdorfer() # Scheme set to Hundsdorfer
    engine = ql.FdHestonVanillaEngine(
        heston_model,
        int(t_grid), int(x_grid), int(v_grid),
        int(damping_steps), scheme_desc
    )

    # 5) Option definition
    payoff = ql.PlainVanillaPayoff(ql_opt_type, float(K))
    exercise = ql.AmericanExercise(todays_date, expiry_date) # American style exercise of the option
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(engine)

    # 6) Price + error handling
    try:
        price = float(option.NPV())
    except:
        price = np.nan

    return price


