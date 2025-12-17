import QuantLib as ql
import numpy as np

class HestonFDPricer:
    """
    Reuses:
      - evaluation date
      - spot quote
      - r/q quotes
      - flat term structures built off QuoteHandles
      - scheme/grid/damping configuration

    Builds per-call:
      - HestonProcess + HestonModel (since params change)
      - FD engine (since model changes)
      - Option payoff/exercise (since K/T change)
    """

    def __init__(
        self,
        todays_date: ql.Date,
        calendar: ql.Calendar,
        day_count: ql.DayCounter,
        engine_grid=(600, 2000, 900),
        damping_steps=8,
        scheme="hundsdorfer",
        spot_init=205.0,
        r_init=0.03,
        q_init=0.01,
        use_calendar_advance=True,
    ):
        self.todays_date = todays_date
        self.calendar = calendar
        self.day_count = day_count
        self.engine_grid = tuple(engine_grid)
        self.damping_steps = int(damping_steps)
        self.use_calendar_advance = use_calendar_advance

        # Anchor QuantLib to valuation date once per process
        ql.Settings.instance().evaluationDate = self.todays_date

        # Quotes we can update each row
        self._spot_q = ql.SimpleQuote(float(spot_init))
        self._r_q = ql.SimpleQuote(float(r_init))
        self._q_q = ql.SimpleQuote(float(q_init))

        spot_handle = ql.QuoteHandle(self._spot_q)
        r_handle = ql.QuoteHandle(self._r_q)
        q_handle = ql.QuoteHandle(self._q_q)

        # Flat term structures driven by quote handles (so updating quotes updates curves)
        self.spot_handle = spot_handle
        self.rf_curve = ql.YieldTermStructureHandle(
            ql.FlatForward(self.todays_date, r_handle, self.day_count)
        )
        self.div_curve = ql.YieldTermStructureHandle(
            ql.FlatForward(self.todays_date, q_handle, self.day_count)
        )

        # FD scheme (fixed)
        scheme = scheme.lower()
        if scheme == "hundsdorfer":
            self.scheme_desc = ql.FdmSchemeDesc.Hundsdorfer()
        elif scheme == "douglas":
            self.scheme_desc = ql.FdmSchemeDesc.Douglas()
        elif scheme in ("cranknicolson", "cn"):
            self.scheme_desc = ql.FdmSchemeDesc.CrankNicolson()
        else:
            raise ValueError("scheme must be 'hundsdorfer', 'douglas', or 'cranknicolson'.")

    def _expiry_date(self, T_years: float) -> ql.Date:
        days = int(max(1, round(float(T_years) * 365.0)))
        if self.use_calendar_advance:
            return self.calendar.advance(self.todays_date, ql.Period(days, ql.Days))
        else:
            return self.todays_date + days

    def set_market(self, r: float, q: float):
        # Update quotes in place (cheap)
        self._r_q.setValue(float(r))
        self._q_q.setValue(float(q))

    def price_american(
        self,
        K: float,
        T: float,
        v0: float,
        theta: float,
        kappa: float,
        sigma: float,
        rho: float,
        option_type="call",
    ) -> float:
        opt = option_type.lower()
        if opt in ("call", "c"):
            ql_opt_type = ql.Option.Call
        elif opt in ("put", "p"):
            ql_opt_type = ql.Option.Put
        else:
            raise ValueError("option_type must be 'call'/'c' or 'put'/'p'")

        expiry_date = self._expiry_date(T)

        # Build Heston process/model per call (params change)
        process = ql.HestonProcess(
            self.rf_curve,
            self.div_curve,
            self.spot_handle,
            float(v0),
            float(kappa),
            float(theta),
            float(sigma),
            float(rho),
        )
        model = ql.HestonModel(process)

        t_grid, x_grid, v_grid = self.engine_grid
        engine = ql.FdHestonVanillaEngine(
            model,
            int(t_grid), int(x_grid), int(v_grid),
            int(self.damping_steps),
            self.scheme_desc,
        )

        payoff = ql.PlainVanillaPayoff(ql_opt_type, float(K))
        exercise = ql.AmericanExercise(self.todays_date, expiry_date)
        option = ql.VanillaOption(payoff, exercise)
        option.setPricingEngine(engine)

        try:
            return float(option.NPV())
        except RuntimeError:
            return np.nan
