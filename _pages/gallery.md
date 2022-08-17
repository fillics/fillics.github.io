---
layout: article
permalink: gallery/
date: 2018-02-18
modified: 2018-03-25
excerpt: "Projects big and small I've been working on"
author_profile: false
comments: false
share: false
---

<div>
{% for post in site.categories.projects %}
  {% include post-grid.html %}
{% endfor %}
</div><!-- /.tiles -->
