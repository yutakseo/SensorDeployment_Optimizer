def run(
    self,
    selection_method: str = "elite",
    tournament_size: int = 3,
    mutation_rate: float = 0.7,
    verbose: bool = True,
    profile: bool = True,
    profile_every: int = 1,
    profile_fitness_breakdown: bool = True,
    # -------------------------
    # NEW: Early stopping knobs
    # -------------------------
    early_stop: bool = True,
    early_stop_coverage: float = 90.0,
    early_stop_patience: int = 10,
) -> Generation:
    population = self.population

    log_eval = FitnessFunc(
        jobsite_map=self.jobsite_map,
        corner_positions=self.corner_positions,
        coverage=self.coverage,
        **self.fitness_kwargs,
    )

    # -------------------------
    # NEW: Early stopping state
    # -------------------------
    stable_count = 0
    last_best_total: Optional[int] = None

    for gen_idx in range(1, self.generations + 1):
        prof: Dict[str, float] = {}

        with _timer("fitness_total", prof):
            sorted_generation, fitness_scores = self.fitness(
                population,
                profile_acc=prof if profile else None,
                profile_breakdown=bool(profile and profile_fitness_breakdown),
            )

        if not fitness_scores:
            break

        # ====== best stats 계산 (verbose와 무관하게 early stop에 필요) ======
        corner_cnt = len(self.corner_positions)
        total_counts = [corner_cnt + len(ch) for ch in sorted_generation]

        best_inner = sorted_generation[0]
        _, best_cov, best_total = log_eval.evaluate(best_inner)  # best_total_sensors

        best_fitness = fitness_scores[0]
        worst_fitness = fitness_scores[-1]
        avg_fitness = sum(fitness_scores) / len(fitness_scores)

        if verbose:
            self._log_generation(
                gen_idx,
                best_fitness=best_fitness,
                avg_fitness=avg_fitness,
                worst_fitness=worst_fitness,
                best_coverage=best_cov,
                corner_sensor_count=corner_cnt,
                sensors_min=min(total_counts),
                sensors_avg=sum(total_counts) / len(total_counts),
                sensors_max=max(total_counts),
                best_total_sensors=best_total,
            )

        # -------------------------
        # NEW: Early stopping check
        # -------------------------
        if early_stop and (best_cov >= float(early_stop_coverage)):
            if last_best_total is None:
                last_best_total = best_total
                stable_count = 1
            else:
                if best_total == last_best_total:
                    stable_count += 1
                else:
                    last_best_total = best_total
                    stable_count = 1  # "현재 세대부터" 다시 카운트

            if stable_count >= int(early_stop_patience):
                if verbose:
                    print(
                        f"[EarlyStop] Gen={gen_idx:03d} | "
                        f"Coverage(best)={best_cov:.2f}% >= {early_stop_coverage:.2f}% and "
                        f"BestSensors={best_total} stable for {stable_count} generations."
                    )
                break
        else:
            # coverage 조건이 깨지면 안정 카운트 리셋
            stable_count = 0
            last_best_total = None

        with _timer("selection_total", prof):
            parents = self.selection(
                sorted_generation,
                fitness_scores,
                method=selection_method,
                tournament_size=tournament_size,
            )

        if len(parents) < 2:
            break

        children: Generation = []
        with _timer("reproduction_total", prof):
            while len(children) < self.child_size:
                p1, p2 = random.sample(parents, 2)

                with _timer("crossover_total", prof):
                    child = self.crossover(p1, p2)

                if mutation_rate > 0 and random.random() < float(mutation_rate):
                    with _timer("mutation_total", prof):
                        child = self.mutation(child)

                children.append(child)

        population = children

        if profile and (gen_idx % int(profile_every) == 0):
            self._log_profile(gen_idx, prof, child_size=self.child_size, mutation_rate=float(mutation_rate))

    self.population = population
    return self.population
