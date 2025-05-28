use crate::track::{ObservationAttributes, ObservationMetricOk};
use crate::voting::topn::TopNVotingElt;
use crate::voting::Voting;
use itertools::Itertools;
use log::debug;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

/// TopN winners voting engine that selects Top N vectors with most close distances.
///
/// It calculates winners as:
/// 1. removes all distances that are greater than threshold
/// 2. sorts remaining tracks according to their IDs
/// 3. counts tracks by their ID's
/// 4. sorts groups by frequency decreasingly
/// 5. returns TopN
///
pub struct BestFitVoting<OA>
where
    OA: ObservationAttributes,
{
    max_distance: f32,
    min_votes: usize,
    _phony: PhantomData<OA>,
}

impl<OA> BestFitVoting<OA>
where
    OA: ObservationAttributes,
{
    /// Constructs new engine
    ///
    /// # Arguments
    /// * `max_distance` - max distance permitted to participate
    /// * `min_votes` - minimal amount of votes required the track to participate
    ///
    pub fn new(max_distance: f32, min_votes: usize) -> Self {
        Self {
            max_distance,
            min_votes,
            _phony: PhantomData,
        }
    }
}

impl<OA> Voting<OA> for BestFitVoting<OA>
where
    OA: ObservationAttributes,
{
    type WinnerObject = TopNVotingElt;

    fn winners<T>(&self, distances: T) -> HashMap<u64, Vec<TopNVotingElt>>
    where
        T: IntoIterator<Item = ObservationMetricOk<OA>>,
    {
        let mut max_dist = -1.0_f32;
        let mut candidates: Vec<_> = distances
            .into_iter()
            .filter(
                |ObservationMetricOk {
                     from: q,
                     to: w,
                     attribute_metric: _f_attr_dist,
                     feature_distance: feat_dist,
                 }| {
                    debug!(
                        "Raw | Src: {:#?}, Dst: {:#?}, Metric: {:#?}",
                        q, w, feat_dist
                    );
                    match feat_dist {
                        Some(e) => {
                            if max_dist < *e {
                                max_dist = *e;
                            }
                            *e <= self.max_distance
                        }
                        _ => false,
                    }
                },
            )
            .map(
                |ObservationMetricOk {
                     from: src_track,
                     to: dest_track,
                     attribute_metric: _,
                     feature_distance: dist,
                 }| { ((src_track, dest_track), dist.unwrap()) },
            )
            .into_group_map()
            .into_iter()
            .filter(|(_, count)| count.len() >= self.min_votes)
            .map(|((src_track, dest_track), dists)| {
                debug!(
                    "Group | Src: {:#?}, Dst: {:#?}, Dist: {:#?}",
                    src_track, dest_track, &dists
                );
                let weight = dists.into_iter().map(|d| (max_dist - d) as f64).sum();
                TopNVotingElt {
                    query_track: src_track,
                    winner_track: dest_track,
                    weight,
                }
            })
            .collect::<Vec<_>>();

        candidates.sort_by(|e1, e2| e2.weight.partial_cmp(&e1.weight).unwrap());

        debug!("Candidates: {:#?}", &candidates);

        let mut results: HashSet<u64> = HashSet::new();

        for c in &mut candidates {
            let key = c.query_track;
            let winner = c.winner_track;
            if results.contains(&winner) {
                c.winner_track = key;
            } else {
                results.insert(winner);
            }
        }

        let res = candidates
            .into_iter()
            .map(|e| (e.query_track, e))
            .into_group_map();
        debug!("Results: {:#?}", &res);
        res
    }
}

pub struct BestFitVotingWithFallback<OA>
where
    OA: ObservationAttributes,
{
    pub max_distance: f32,
    pub min_votes: usize,
    _phantom: PhantomData<OA>,
}

impl<OA> BestFitVotingWithFallback<OA>
where
    OA: ObservationAttributes,
{
    pub fn new(max_distance: f32, min_votes: usize) -> Self {
        Self {
            max_distance,
            min_votes,
            _phantom: PhantomData,
        }
    }
}

impl<OA> Voting<OA> for BestFitVotingWithFallback<OA>
where
    OA: ObservationAttributes,
{
    type WinnerObject = TopNVotingElt;

    fn winners<T>(&self, distances: T) -> HashMap<u64, Vec<TopNVotingElt>>
    where
        T: IntoIterator<Item = ObservationMetricOk<OA>>,
    {
        let mut max_dist = -1.0_f32;

        // Step 1: group all distances by (from, to), filter by max_distance
        let grouped: HashMap<(u64, u64), Vec<f32>> = distances
            .into_iter()
            .filter_map(|d| match d.feature_distance {
                Some(f) if f <= self.max_distance => {
                    max_dist = max_dist.max(f);
                    Some(((d.from, d.to), f))
                }
                _ => None,
            })
            .into_group_map();

        // Step 2: filter by min_votes
        let filtered: Vec<TopNVotingElt> = grouped
            .into_iter()
            .filter(|(_, v)| v.len() >= self.min_votes)
            .map(|((from, to), dists)| {
                let weight = dists.into_iter().map(|d| (max_dist - d) as f64).sum();
                TopNVotingElt {
                    query_track: from,
                    winner_track: to,
                    weight,
                }
            })
            .collect();

        // Step 3: group by query (from), and sort each list by descending weight
        let mut per_query = filtered.into_iter().into_group_map_by(|e| e.query_track);

        for candidates in per_query.values_mut() {
            candidates.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap());
        }

        // Step 4: assign each query to its best available winner (fallback to self)
        let mut used_winners = HashSet::new();
        let mut final_map = HashMap::new();

        for (query_id, candidates) in per_query {
            let mut assigned = false;
            for mut cand in candidates {
                if !used_winners.contains(&cand.winner_track) {
                    used_winners.insert(cand.winner_track);
                    final_map.insert(query_id, vec![cand]);
                    assigned = true;
                    break;
                }
            }

            if !assigned {
                final_map.insert(
                    query_id,
                    vec![TopNVotingElt {
                        query_track: query_id,
                        winner_track: query_id,
                        weight: 0.0,
                    }],
                );
            }
        }

        final_map
    }
}