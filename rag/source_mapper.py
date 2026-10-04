import csv
import os
from typing import Dict, List, Optional


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SOURCES_FILE = os.path.join(
    BASE_DIR,
    "data",
    "sources.csv"
)


class SourceMapper:
    """
    Maps source IDs used by the RAG system back to the
    verified records stored in data/sources.csv.
    """

    def __init__(
        self,
        sources_file: str = SOURCES_FILE,
    ):
        self.sources_file = sources_file
        self.sources = self._load_sources()

    def _load_sources(self) -> Dict[str, Dict]:
        """
        Load all source records from sources.csv.
        """

        if not os.path.exists(
            self.sources_file
        ):
            raise FileNotFoundError(
                f"Sources file not found: "
                f"{self.sources_file}"
            )

        sources = {}

        with open(
            self.sources_file,
            "r",
            encoding="utf-8-sig",
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                source_id = (
                    row.get(
                        "source_id",
                        ""
                    )
                    .strip()
                )

                if not source_id:
                    continue

                sources[source_id] = {
                    "source_id": source_id,
                    "category": row.get(
                        "category",
                        ""
                    ).strip(),
                    "source_name": row.get(
                        "source_name",
                        ""
                    ).strip(),
                    "source_url": row.get(
                        "source_url",
                        ""
                    ).strip(),
                    "authority": row.get(
                        "authority",
                        ""
                    ).strip(),
                    "description": row.get(
                        "description",
                        ""
                    ).strip(),
                    "last_verified": row.get(
                        "last_verified",
                        ""
                    ).strip(),
                }

        return sources

    def get_source(
        self,
        source_id: str,
    ) -> Optional[Dict]:
        """
        Return complete source information for a source ID.
        """

        if not source_id:
            return None

        return self.sources.get(
            source_id.strip()
        )

    def get_sources(
        self,
        source_ids: List[str],
    ) -> List[Dict]:
        """
        Return complete information for multiple source IDs.
        """

        results = []

        seen = set()

        for source_id in source_ids:

            source = self.get_source(
                source_id
            )

            if source is None:
                continue

            source_key = source[
                "source_id"
            ]

            if source_key in seen:
                continue

            seen.add(
                source_key
            )

            results.append(
                source
            )

        return results

    def map_results(
        self,
        results: List[Dict],
    ) -> List[Dict]:
        """
        Add complete verified source metadata to retrieved
        RAG results.
        """

        mapped_results = []

        for result in results:

            result_copy = result.copy()

            metadata = result_copy.get(
                "metadata",
                {}
            ).copy()

            source_id = metadata.get(
                "source_id",
                ""
            )

            source = self.get_source(
                source_id
            )

            if source:

                metadata["source_name"] = source[
                    "source_name"
                ]

                metadata["category"] = source[
                    "category"
                ]

                metadata["authority"] = source[
                    "authority"
                ]

                metadata["source_url"] = source[
                    "source_url"
                ]

                metadata["description"] = source[
                    "description"
                ]

                metadata["last_verified"] = source[
                    "last_verified"
                ]

            result_copy["metadata"] = metadata

            mapped_results.append(
                result_copy
            )

        return mapped_results

    def format_citation(
        self,
        source_id: str,
    ) -> str:
        """
        Create a human-readable citation for a source.
        """

        source = self.get_source(
            source_id
        )

        if not source:
            return (
                f"Source {source_id} "
                f"could not be found."
            )

        return (
            f"{source['source_name']} "
            f"({source['authority']}) — "
            f"{source['source_url']}"
        )


def get_source_by_id(
    source_id: str,
) -> Optional[Dict]:
    """
    Convenience function for retrieving a source.
    """

    mapper = SourceMapper()

    return mapper.get_source(
        source_id
    )


if __name__ == "__main__":

    mapper = SourceMapper()

    print(
        f"Loaded {len(mapper.sources)} "
        f"verified sources."
    )

    # Demonstration using the first available source.
    if mapper.sources:

        first_source_id = next(
            iter(mapper.sources)
        )

        source = mapper.get_source(
            first_source_id
        )

        print("\nExample source:")
        print(
            f"ID: {source['source_id']}"
        )
        print(
            f"Name: {source['source_name']}"
        )
        print(
            f"Authority: {source['authority']}"
        )
        print(
            f"URL: {source['source_url']}"
        )
