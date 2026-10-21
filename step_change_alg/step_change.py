from benchmark.eval.model import Event
import numpy as np
import pandas as pd
from collections import deque
from sklearn.cluster import KMeans
from sklearn import metrics

keys = ["country_code", "date"]

def detect(media_df, sales_df, sid: str) -> list[Event]:
    DEBUG = False
    EPSILON = 10
    NR_OF_CONFIRMATION_DAYS = 7
    MOVING_AVG_WINDOW_SIZE = 15
    events = []

    spend = (
        media_df.groupby(keys + ["advertising_channel"])["media_investment"]
        .sum(min_count=1)
        .unstack()
    )

    sales = sales_df.groupby(keys)["turnover"].sum(min_count=1).rename("total_turnover")

    df = spend.join(sales, how="outer").sort_index().reset_index()
    df.columns.name = None

    df["date"] = pd.to_datetime(df["date"])
    #print(df.columns)

    spend_channels = df.columns.drop(["country_code", "date", "total_turnover"])

    for country in df["country_code"].unique():
        country_data = df[df["country_code"] == country]
        date = country_data["date"]
        last_date = date.max()
        for channel in spend_channels:
            # Making sure that the moving average doesn't blur the
            spend = country_data[[channel]].copy()
            zero_values = spend <= EPSILON
            originial_vals = spend[zero_values]
            spend[zero_values] = np.nan
            spend = spend.rolling(window=MOVING_AVG_WINDOW_SIZE, center=True, min_periods=1).mean()
            spend[zero_values] = originial_vals

            silhouette_index = []
            kmeans_labels = []
            cluster_centers = []
            greatest_score = -1
            greatest_score_n_clusters = 0
            k = 2
            while k <= 40:
                kmeans = KMeans(n_clusters=k, n_init=10).fit(spend)
                try:
                    score = metrics.silhouette_score(spend, kmeans.labels_)
                except:
                    break
                silhouette_index.append(score)
                kmeans_labels.append(kmeans.labels_)
                cluster_centers.append(kmeans.cluster_centers_)
                if score > greatest_score * 1.05:
                    greatest_score = score
                    greatest_score_n_clusters = k
                if k - greatest_score_n_clusters >= 4:
                    break
                k += 1

            if greatest_score >= 0.7:  # There could be a step change
                zero_clusters = set()
                step_changes = []
                start, end = None, None  # The current considered start and end times of the step change as indices of the spend column
                old_label = None  # The label before the step change occurred
                current_label = None
                best_labels = kmeans_labels[greatest_score_n_clusters - 2]
                best_cluster_centers = cluster_centers[
                    greatest_score_n_clusters - 2]  # -2 here because the first iteration is done with 2 clusters
                cluster_counts = dict()
                cluster_history = deque()
                ZERO_THRESHOLD = np.median(spend[(spend > EPSILON).values])/8 # Below this threshold spending is considered to be paused and not a step change

                for label in range(greatest_score_n_clusters):
                    cluster_points = spend.values[best_labels == label]
                    if np.quantile(cluster_points, 0.2) < ZERO_THRESHOLD:
                        zero_clusters.add(label)

                for (index, label) in enumerate(best_labels):
                    label = int(label)
                    cluster_history.append(label)
                    if label in cluster_counts:
                        cluster_counts[label] += 1
                    else:
                        cluster_counts[label] = 1
                    if index >= NR_OF_CONFIRMATION_DAYS:  # the last NR_OF_CONFIRMATION_DAYS clusters are considered at a time
                        cluster_counts[cluster_history.popleft()] -= 1

                    if (current_label is None):
                        if index + 1 >= NR_OF_CONFIRMATION_DAYS:
                            most_freq_label = max(cluster_counts, key=cluster_counts.get)
                            if cluster_counts[most_freq_label] / NR_OF_CONFIRMATION_DAYS > 0.85 and (
                                    most_freq_label not in zero_clusters):
                                current_label = most_freq_label
                    else:  # if there was a current label
                        most_freq_label = max(cluster_counts, key=cluster_counts.get)
                        if cluster_counts[most_freq_label] / NR_OF_CONFIRMATION_DAYS > 0.85:
                            candidate_label = most_freq_label
                            if candidate_label != current_label:  # Here it is identified that a step change has occured
                                for j in range(
                                        NR_OF_CONFIRMATION_DAYS):  # Now we want to determine when was the first day that had spending in the new cluster
                                    if cluster_history[j] == candidate_label:
                                        point = index - NR_OF_CONFIRMATION_DAYS + (
                                                    j + 1)  # this is when the step change period is apparent
                                        break

                                # determining if the point is at the start or end of a step change:
                                if start is None and end is None:  # start needs to be set
                                    if candidate_label in zero_clusters:
                                        pass
                                    else:
                                        start = point
                                        old_label = current_label
                                        current_label = candidate_label

                                elif start != None and end is None:  # end needs to be set
                                    if candidate_label in zero_clusters:
                                        pass
                                    else:  # we have found the end of the step change and the start of the (possible) new one
                                        end = point - 1
                                        step_changes.append([start, end])

                                        if candidate_label != old_label:
                                            start = point
                                            end = None
                                        else:
                                            start = None
                                            end = None
                                        current_label = candidate_label
                if start is not None and end is None:
                    step_changes.append([start, None])

                step_changes = [
                    [date.iloc[period[0]], date.iloc[period[1]]] if not period[1] is None else [date.iloc[period[0]],
                                                                                                last_date] for period in
                    step_changes]
                if DEBUG:
                    print(country, channel)
                    print(step_changes)
                    print("k: ", greatest_score_n_clusters)
                    print("cluster centres: ", best_cluster_centers)
                    print()
                for period in step_changes:
                    events.append(
                        Event(
                            sid=sid,
                            country_code=country,
                            channel=channel,
                            event_type="step_change",
                            start=period[0],
                            end=period[1]

                        )
                    )


            else:  # There is likely just one cluster => no step change at all for this market's channel
                pass
    return events