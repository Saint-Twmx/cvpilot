json = \
    {
        "mitralCalculation": {
            "parameters": [
            ]
        },
        "annularPlane":{
            "markups": [
                {
                    "type":"ClosedCurve",         #闭口曲线
                    "controlPoints":[],           # 二尖瓣瓣环平面点集,36个点
                    "description":"三维二尖瓣瓣环平面,36个点,闭口曲线",
                    "measurements": [
                        {
                            "name": "length",
                            "value": 0,
                            "description": "瓣环平面36个点粗略计算的周长"
                        },
                        {
                            "name": "area",
                            "value": 0,
                            "description": "瓣环平面36个点粗略计算的面积"
                        }
                    ]
                },
            ]
        },

        "annularPlaneProj":{
            "markups": [
                {
                    "type":"ClosedCurve",         #闭口曲线
                    "controlPoints":[],           # 二尖瓣瓣环投影平面点集
                    "description":"二尖瓣瓣环平面基准平面,即投影面,36个点,闭口曲线",
                    "measurements": [
                        {
                            "name": "length",
                            "value": 0,
                            "description": "瓣环投影平面36个点粗略计算的周长"
                        },
                        {
                            "name": "area",
                            "value": 0,
                            "description": "瓣环投影平面36个点粗略计算的面积"
                        }
                    ]
                }
            ]
        },

        "annularHeight": {
            "markups": [
                {
                    "type": "Line",       # 直线
                    "controlPoints": [],
                    "description": "二尖瓣瓣环高度,不是二尖瓣高度,两个点组成一条直线",
                    "measurements": [
                        {
                            "name": "length",
                            "value": 0,
                            "description": "二尖瓣瓣环高度值",
                        }
                    ]
                }
            ]
        },

        #cc  ap  cc_real tt  四个指标
        "cc":{
            "markups": [
                {
                    "type":"Line",                 #直线
                    "controlPoints":[],            #两个点（类长径）
                    "description": "由annularPlane计算得到的二尖瓣CC径,两个点组成一条直线",
                    "measurements": [
                        {
                            "name": "length",
                            "value": 0,
                            "description": "二尖瓣CC径值",
                        }
                    ]                               #三维空间下cc值（空间两个点距离）
                }
            ]
        },

        "ccProj": {
            "markups": [
                {
                    "type": "Line",       # 直线
                    "controlPoints": [],  #两个投影点
                    "description": "由annularPlaneProj计算得到的二尖瓣CC径,两个点组成一条直线",
                    "measurements": [     #投影平面下cc值（平面两个点距离）
                        {
                            "name": "length",
                            "value": 0,
                            "description": "二尖瓣CC径投影值",
                        }
                    ]
                }
            ]
        },

        "ccReal": {
            "markups": [
                {
                    "type": "Line",                 # 直线
                    "controlPoints": [],            #前后叶交接两个点
                    "description": "由annularPlane计算得到的二尖瓣CCReal径,两个点组成一条直线",
                    "measurements": [               # 三维空间下real cc值（空间两个点距离）
                        {
                            "name": "length",
                            "value": 0,
                            "description": "二尖瓣CCReal径值",
                        }
                    ]
                }
            ]
        },

        "ccRealProj": {
            "markups": [
                {
                    "type": "Line",             # 直线
                    "controlPoints": [],
                    "description": "由annularPlaneProj计算得到的二尖瓣CCReal径,两个点组成一条直线",
                    "measurements": [           #投影平面下real cc值（平面两个点距离）
                        {
                            "name": "length",
                            "value": 0,
                            "description": "二尖瓣CCReal径投影值",
                        }
                    ]
                }
            ]
        },

        "ap": {
            "markups": [
                {
                    "type": "Line",                 # 直线
                    "controlPoints": [],            #前后叶中点两个点（类短径）
                    "description": "由annularPlane计算得到的二尖瓣AP径,两个点组成一条直线",
                    "measurements": [               #三维空间下ap值（空间两个点距离）
                        {
                            "name": "length",
                            "value": 0,
                            "description": "二尖瓣AP径值",
                        }
                    ]
                }
            ]
        },

        "apProj": {
            "markups": [
                {
                    "type": "Line",                 # 直线
                    "controlPoints": [],
                    "description": "由annularPlaneProj计算得到的二尖瓣AP径,两个点组成一条直线",
                    "measurements": [               #投影平面下ap值（平面两个点距离）
                        {
                            "name": "length",
                            "value": 0,
                            "description": "二尖瓣AP径投影值",
                        }
                    ]
                }
            ]
        },

        "tt": {
            "markups": [
                {
                    "type": "Line",                 # 直线
                    "controlPoints": [],            # TT两个点
                    "description": "由annularPlane计算得到的二尖瓣TT径,两个点组成一条直线",
                    "measurements": [               #两个点对应距离
                        {
                            "name": "length",
                            "value": 0,
                            "description": "二尖瓣TT径值",
                        }
                    ]
                }
            ]
        },

        "a1":{
            "markups":[
                {
                    "type": "Curve",                  #开口曲线
                    "controlPoints":[],              #瓣叶长度点
                    "description": "计算得到的二尖瓣A1区瓣叶长度控制点,多个点,开口曲线,A23/P123均是如此不再重复",
                    "measurements": [
                        {
                            "name": "line_lenght",      #瓣叶长度直线距离
                            "value": 0,
                            "description": "A1区瓣叶首尾两个点直线距离,A23/P123均是如此不再重复",
                        },
                        {
                            "name": "curve_lenght",     #瓣叶长度曲线距离
                            "value": 0,
                            "description": "A1区瓣叶所有控制点曲线距离,A23/P123均是如此不再重复",
                        }
                    ]
                }
            ]
        },
        "a2":{
            "markups":[
                {
                    "type": "Curve",                  #开口曲线
                    "controlPoints":[],
                    "measurements": [
                        {
                            "name": "line_lenght",      #瓣叶长度直线距离
                            "value": 0,
                        },
                        {
                            "name": "curve_lenght",     #瓣叶长度曲线距离
                            "value": 0,
                        }
                    ]
                }
            ]
        },
        "a3":{
            "markups":[
                {
                    "type": "Curve",                  #开口曲线
                    "controlPoints":[],
                    "measurements": [
                        {
                            "name": "line_lenght",      #瓣叶长度直线距离
                            "value": 0,
                        },
                        {
                            "name": "curve_lenght",     #瓣叶长度曲线距离
                            "value": 0,
                        }
                    ]
                }
            ]
        },
        "p1":{
            "markups":[
                {
                    "type": "Curve",                  #开口曲线
                    "controlPoints":[],
                    "measurements": [
                        {
                            "name": "line_lenght",      #瓣叶长度直线距离
                            "value": 0,
                        },
                        {
                            "name": "curve_lenght",     #瓣叶长度曲线距离
                            "value": 0,
                        }
                    ]
                }
            ]
        },
        "p2":{
            "markups":[
                {
                    "type": "Curve",                  #开口曲线
                    "controlPoints":[],
                    "measurements": [
                        {
                            "name": "line_lenght",      #瓣叶长度直线距离
                            "value": 0,
                        },
                        {
                            "name": "curve_lenght",     #瓣叶长度曲线距离
                            "value": 0,
                        }
                    ]
                }
            ]
        },
        "p3":{
            "markups":[
                {
                    "type": "Curve",                  #开口曲线
                    "controlPoints":[],
                    "measurements": [
                        {
                            "name": "line_lenght",      #瓣叶长度直线距离
                            "value": 0,
                        },
                        {
                            "name": "curve_lenght",     #瓣叶长度曲线距离
                            "value": 0,
                        }
                    ]
                }
            ]
        }
    }