/*
 * simulation.js
 * ------------------------------------------------------------------
 * Faithful JavaScript port of the simulation math from the original
 * Streamlit app (single_sim.py + multiple_sims.py).
 *
 * Every deterministic function here is numerically identical (to IEEE-754
 * double precision) to its Python/numpy counterpart. The only difference is
 * the random number generator used for Monte-Carlo sampling in the multiple
 * simulations mode (JavaScript's Math.random instead of numpy) — the sampling
 * *distributions* and all downstream logic are identical.
 *
 * Works both in the browser (attaches to window.Sim) and in Node (module.exports).
 */
(function (root) {
  'use strict';

  // ---- numpy-equivalent helpers -----------------------------------------
  const log = Math.log;
  const log2 = Math.log2;
  const exp = Math.exp;
  const abs = Math.abs;

  // Python int() truncates toward zero (inputs here are always >= 0).
  function toInt(x) { return Math.trunc(x); }

  // numpy.linspace(start, stop, num) inclusive of both endpoints.
  function linspace(start, stop, num) {
    const out = [];
    if (num === 1) { return [start]; }
    const step = (stop - start) / (num - 1);
    for (let i = 0; i < num; i++) out.push(start + step * i);
    return out;
  }

  // ======================================================================
  // SINGLE SIMULATION  (port of single_sim.py)
  // ======================================================================

  // transform_sizes_to_years: 256^n -> n  (years of progress at recent pace)
  function transformSizesToYears(sizes, softwareContribution) {
    const softwareDoublesPerYear = 4;
    const normalizer = softwareDoublesPerYear / softwareContribution;
    return sizes.map(function (size) { return log2(size) / normalizer; });
  }

  /*
   * runSingle(params) -> full result object for the single simulation.
   * params: {
   *   f, r0, yrLeft, lambda, softwareContribution,
   *   computeGrowth (bool), retrainingCost (bool), constantR (bool),
   *   fMin, fMax    (used only when computeGrowth === true)
   * }
   */
  function runSingle(p) {
    const computeGrowth = !!p.computeGrowth;
    const retrainingCost = !!p.retrainingCost;
    const constantR = !!p.constantR;
    const softwareContributionParam = p.softwareContribution;

    // --- choose_parameters() ---
    const factorIncrease = computeGrowth ? 1.1 : 2;
    const rInitial = p.r0;
    let f0, fMax;
    if (computeGrowth) {
      f0 = p.fMin;
      fMax = p.fMax;
    } else {
      f0 = p.f;
      fMax = p.f;
    }
    const computeSizeStart = 1;
    const computeMax = 4096;
    const computeDoublingTime = 5;
    const computeGrowthMonthlyRate = log(2) / computeDoublingTime;
    const limitYears = p.yrLeft;
    const softwareContribution = softwareContributionParam;
    const lambdaFactor = p.lambda;
    const doublingTimeStarting = 3;
    const impliedMonthGrowthRate = log(2) / doublingTimeStarting;
    const timeTakesToFactorIncrease = log(factorIncrease) / impliedMonthGrowthRate;
    const initialFactorIncreaseTime = timeTakesToFactorIncrease / (1 + f0);

    // --- dynamic_system() ---
    const maxTimeMonths = 72;
    const ceiling = Math.pow(Math.pow(2, 4 / softwareContribution), limitYears);
    let r = rInitial;
    let factorIncreaseTime = initialFactorIncreaseTime;
    let size = 1.0;
    let computeSize = computeSizeStart;

    const times = [0];
    const sizes = [size];
    const rs = [r];
    const computeSizes = [computeSize];
    const fValues = [f0];
    let f = f0;
    const totalFactorIncreasings = log(ceiling) / log(factorIncrease);
    const k = constantR ? 0 : rInitial / totalFactorIncreasings;

    let timeElapsed = 0;
    while (timeElapsed < maxTimeMonths && size < ceiling && r > 0) {
      const dt = Math.min(factorIncreaseTime, maxTimeMonths - timeElapsed);
      const fOld = f;

      timeElapsed += dt;
      times.push(timeElapsed);

      size *= Math.pow(factorIncrease, dt / factorIncreaseTime);
      if (size > ceiling) size = ceiling;
      sizes.push(size);

      r -= k * (dt / factorIncreaseTime);
      rs.push(r);

      computeSize = computeSizeStart * exp(computeGrowthMonthlyRate * timeElapsed);
      computeSizes.push(computeSize);

      if (computeSize < computeMax) {
        const fGrowthRate = log(fMax / f0) / (5 * 12);
        f = f0 * exp(timeElapsed * fGrowthRate);
      } else {
        f = fMax;
      }
      fValues.push(f);

      if (r > 0) {
        let accelFactor;
        if (retrainingCost) {
          accelFactor = (lambdaFactor * ((1 / r) - 1)) /
                        (abs(lambdaFactor * ((1 / r) - 1)) + 1);
        } else {
          accelFactor = lambdaFactor * (1 / r - 1);
        }
        factorIncreaseTime *= (Math.pow(factorIncrease, accelFactor) / ((1 + f) / (1 + fOld)));
      }
    }

    // --- derived series (exactly as plotted in single_sim.py) ---
    const transformedSizes = transformSizesToYears(sizes, softwareContributionParam);
    const timesInYears = times.map(function (t) { return t / 12; });

    const growthRatesFull = [];
    for (let i = 1; i < sizes.length; i++) {
      const dt = timesInYears[i] - timesInYears[i - 1];
      if (dt > 0) {
        growthRatesFull.push((sizes[i] - sizes[i - 1]) / (sizes[i] * dt));
      } else {
        growthRatesFull.push(NaN);
      }
    }
    // growth_times = times_in_years[1:] ; drop last obs from both
    let growthTimes = timesInYears.slice(1);
    growthTimes = growthTimes.slice(0, -1);
    const growthRates = growthRatesFull.slice(0, -1);

    return {
      times: times, sizes: sizes, rs: rs, ceiling: ceiling,
      compute_sizes: computeSizes, f_values: fValues,
      transformed_sizes: transformedSizes, times_in_years: timesInYears,
      growth_times: growthTimes, growth_rates: growthRates
    };
  }

  // ======================================================================
  // MULTIPLE SIMULATIONS  (port of multiple_sims.py)
  // ======================================================================

  // uniform draw on [a, b)
  function uniform(a, b) { return a + Math.random() * (b - a); }

  // dynamic_system_with_lambda
  function dynamicSystemWithLambda(rInitial, factorIncrease, initialFactorIncreaseTime,
                                   limitYears, computeGrowthMonthlyRate, f0, fMax,
                                   lambdaFactor, softwareContributionParam,
                                   retrainingCost, constantR, maxTimeMonths) {
    if (maxTimeMonths === undefined) maxTimeMonths = 48;
    const ceiling = Math.pow(Math.pow(2, 4 / softwareContributionParam), limitYears);
    let size = 1.0;
    let r = rInitial;
    let f = f0;

    const times = [0], sizes = [size], rs = [r], computeSizes = [1], fValues = [f];
    let timeElapsed = 0;
    const k = constantR ? 0 : rInitial / (log(ceiling) / log(factorIncrease));

    while (timeElapsed < maxTimeMonths && size < ceiling && r > 0) {
      const fOld = f;
      timeElapsed += initialFactorIncreaseTime;
      size *= factorIncrease;
      const computeSize = computeSizes[computeSizes.length - 1] *
                          exp(computeGrowthMonthlyRate * initialFactorIncreaseTime);

      if (computeSize < 4096) {
        const fGrowthRate = log(fMax / f0) / (5 * 12);
        f = f0 * exp(timeElapsed * fGrowthRate);
      } else {
        f = fMax;
      }

      r -= k;
      times.push(timeElapsed);
      sizes.push(size);
      rs.push(r);
      computeSizes.push(computeSize);
      fValues.push(f);
      if (r > 0) {
        const accelFactor = retrainingCost
          ? (lambdaFactor * ((1 / r) - 1)) / (abs(lambdaFactor * ((1 / r) - 1) + 1))
          : lambdaFactor * (1 / r - 1);
        initialFactorIncreaseTime *= (Math.pow(factorIncrease, accelFactor) / ((1 + f) / (1 + fOld)));
      }
    }
    return { times: times, sizes: sizes, rs: rs, compute_sizes: computeSizes, f_values: fValues };
  }

  // sample_parameters_batch -> array of param objects (one per simulation)
  function sampleParametersBatch(nSamples, bounds, softwareContributionParam, computeGrowth) {
    const factorIncrease = 2;
    const computeGrowthMonthlyRate = log(2) / 5;
    const impliedMonthGrowthRate = log(2) / 3;
    const timeTakesToFactorIncrease = log(factorIncrease) / impliedMonthGrowthRate; // = 3
    const params = [];
    for (let i = 0; i < nSamples; i++) {
      const initialBoost = exp(uniform(log(bounds.ib_low), log(bounds.ib_high)));
      const rInitial = exp(uniform(log(bounds.r_low), log(bounds.r_high)));
      const limitYears = uniform(bounds.ly_low, bounds.ly_high);
      const lambdaFactor = exp(uniform(log(bounds.lf_low), log(bounds.lf_high)));
      const denominator = computeGrowth ? 1.1 : initialBoost;
      const initialFactorIncreaseTime = timeTakesToFactorIncrease / denominator;
      const f0 = computeGrowth ? 0.1 : initialBoost;
      const fMax = initialBoost;
      params.push({
        r_initial: rInitial,
        factor_increase: factorIncrease,
        initial_factor_increase_time: initialFactorIncreaseTime,
        limit_years: limitYears,
        compute_growth_monthly_rate: computeGrowthMonthlyRate,
        f_0: f0,
        f_max: fMax,
        lambda_factor: lambdaFactor,
        software_contribution_param: softwareContributionParam
      });
    }
    return params;
  }

  // calculate_summary_statistics_binary
  // conditions: array of [time_period, speed_up_factor]
  function calculateSummaryStatisticsBinary(times, conditions, softwareContributionParam) {
    const results = {};
    for (let c = 0; c < conditions.length; c++) {
      const timePeriod = conditions[c][0];
      const speedUpFactor = conditions[c][1];
      const baselineDoublings = (timePeriod / 12) * (4 / softwareContributionParam);
      const requiredDoublings = toInt(baselineDoublings * speedUpFactor);
      let hit = 'no';
      for (let i = 0; i < times.length - requiredDoublings; i++) {
        const timeSpan = times[i + requiredDoublings] - times[i];
        if (timeSpan < timePeriod) { hit = 'yes'; break; }
      }
      results[conditions[c][0] + '_' + conditions[c][1]] = hit;
    }
    return results;
  }

  // Helper used by the CDF: fraction of sims that compress `doublings` steps
  // into < time_period_months at some point along their trajectory.
  function successCountForDoublings(timesMatrix, doublings, timePeriodMonths) {
    let successCount = 0;
    for (let m = 0; m < timesMatrix.length; m++) {
      const times = timesMatrix[m];
      if (doublings < times.length) {
        let achieved = false;
        for (let i = 0; i < times.length - doublings; i++) {
          if (times[i + doublings] - times[i] < timePeriodMonths) { achieved = true; break; }
        }
        if (achieved) successCount++;
      }
    }
    return successCount;
  }

  // calculate_years_compressed_cdf
  function calculateYearsCompressedCdf(timesMatrix, softwareContributionParam,
                                       timePeriodMonths, maxYears, resolution) {
    if (timePeriodMonths === undefined) timePeriodMonths = 12;
    if (maxYears === undefined) maxYears = 20;
    if (resolution === undefined) resolution = 200;
    const yearsPoints = linspace(0.1, maxYears, resolution);
    const fractions = [];
    const n = timesMatrix.length;
    for (let j = 0; j < yearsPoints.length; j++) {
      const yearsToCompress = yearsPoints[j];
      const speedUpFactor = yearsToCompress / (timePeriodMonths / 12);
      const baselineDoublings = (timePeriodMonths / 12) * (4 / softwareContributionParam);
      const requiredDoublings = baselineDoublings * speedUpFactor;
      const doublingsFloor = toInt(Math.floor(requiredDoublings));
      const doublingsCeil = toInt(Math.ceil(requiredDoublings));
      let fraction;
      if (doublingsFloor === doublingsCeil) {
        fraction = successCountForDoublings(timesMatrix, doublingsFloor, timePeriodMonths) / n;
      } else {
        const floorCount = successCountForDoublings(timesMatrix, doublingsFloor, timePeriodMonths);
        const ceilCount = successCountForDoublings(timesMatrix, doublingsCeil, timePeriodMonths);
        const weight = requiredDoublings - doublingsFloor;
        fraction = ((1 - weight) * floorCount + weight * ceilCount) / n;
      }
      fractions.push(fraction);
    }
    return { years: yearsPoints, fractions: fractions };
  }

  /*
   * runMultiple(nSims, bounds, softwareContributionParam, flags, onProgress)
   * Returns { probabilities, timesMatrix, sizesMatrix, paramsList }.
   * conditions match the original: [(12,3),(4,10),(12,10),(4,30)].
   */
  function runMultiple(nSims, bounds, softwareContributionParam, flags, onProgress) {
    const conditions = [[12, 3], [4, 10], [12, 10], [4, 30]];
    const paramsBatch = sampleParametersBatch(nSims, bounds, softwareContributionParam, flags.computeGrowth);
    const timesMatrix = [];
    const sizesMatrix = [];
    const paramsList = [];

    for (let i = 0; i < paramsBatch.length; i++) {
      const pr = paramsBatch[i];
      const res = dynamicSystemWithLambda(
        pr.r_initial, pr.factor_increase, pr.initial_factor_increase_time,
        pr.limit_years, pr.compute_growth_monthly_rate, pr.f_0, pr.f_max,
        pr.lambda_factor, pr.software_contribution_param,
        flags.retrainingCost, flags.constantR);
      timesMatrix.push(res.times);
      sizesMatrix.push(res.sizes);
      paramsList.push(pr);
      if (onProgress && (i % 200 === 0 || i === paramsBatch.length - 1)) {
        onProgress((i + 1) / nSims);
      }
    }

    // batch_summary -> probabilities
    const batchSummary = {};
    conditions.forEach(function (c) { batchSummary[c[0] + '_' + c[1]] = 0; });
    for (let m = 0; m < timesMatrix.length; m++) {
      const stats = calculateSummaryStatisticsBinary(timesMatrix[m], conditions, softwareContributionParam);
      conditions.forEach(function (c) {
        if (stats[c[0] + '_' + c[1]] === 'yes') batchSummary[c[0] + '_' + c[1]] += 1;
      });
    }
    const probabilities = {};
    conditions.forEach(function (c) {
      probabilities[c[0] + '_' + c[1]] = batchSummary[c[0] + '_' + c[1]] / nSims;
    });

    return { probabilities: probabilities, timesMatrix: timesMatrix,
             sizesMatrix: sizesMatrix, paramsList: paramsList };
  }

  // ----------------------------------------------------------------------
  const Sim = {
    // single
    runSingle: runSingle,
    transformSizesToYears: transformSizesToYears,
    // multiple
    dynamicSystemWithLambda: dynamicSystemWithLambda,
    sampleParametersBatch: sampleParametersBatch,
    calculateSummaryStatisticsBinary: calculateSummaryStatisticsBinary,
    calculateYearsCompressedCdf: calculateYearsCompressedCdf,
    runMultiple: runMultiple,
    // helpers
    linspace: linspace
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = Sim;
  } else {
    root.Sim = Sim;
  }
})(typeof window !== 'undefined' ? window : this);
